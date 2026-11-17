"""Bridge between TRXViz (Rust GUI) and MPAT + MITTENS (Python).

Provides two subcommands invoked by TRXViz's background worker pool:
1. ``python -m mpat.trxviz_bridge build_transitions <request.json>``
   Computes MITTENS v2 transition probability summary volumes (``codi_double``,
   ``codi_single``, ``coasy``, ``p_terminate_double``, ``p_terminate_single``,
   ``total_amplitude``, ``kl_single``) and builds/caches the 26-class MPAT
   absorbing Markov chain, seed launches, fixel lobes, and whole-brain reference
   density (plus ``chain_dead.nii``).

2. ``python -m mpat.trxviz_bridge solve_bundle <request.json>``
   Loads the cached transition state, applies pyAFQ-style ROIs plus arbitrary
   user-wired Waypoint (``include``), Avoid (``exclude``), Limiting, Endpoint
   (``start``/``end``), Absorption (``term_mask`` — including interior white-matter
   absorption!), and No-End (``no_end_mask``) regions, solves the absorbing Markov
   chain, computes all 7 quantitative 3-D volumes (``afd``, ``density``, ``tdimm``,
   ``qharmonic``, ``coherence``, ``keff``, ``endpoints``), and extracts the 3-D
   reactive-flux streamlines and branch arbor into ``bundle.trx``.

Progress is streamed to stdout as ``PROGRESS <done> <total>`` lines so TRXViz can
render a live progress bar on the active workflow node.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


def _emit_progress(done: int, total: int) -> None:
    print(f"PROGRESS {int(done)} {int(total)}", flush=True)


class LiveDensityStreamer:
    """Double-buffered memory-mapped 3-D volume streamer between MPAT and TRXViz."""

    HEADER_BYTES = 128

    def __init__(self, mmap_path: Path, chain, affine: np.ndarray, min_interval_sec: float = 0.04) -> None:
        import struct

        self._struct = struct
        self.path = Path(mmap_path)
        self.shape = tuple(int(d) for d in chain.shape)
        self.nvox = int(np.prod(self.shape))
        self.slot_bytes = self.nvox * 4
        self.total_bytes = self.HEADER_BYTES + 2 * self.slot_bytes
        self.affine_f32 = np.asarray(affine, dtype=np.float32).reshape(16)
        self.t_ijk = tuple((np.stack(np.unravel_index(chain.t_lin, chain.pshape), 1) - 1).T)
        self.seq = 0
        self.min_interval_sec = float(min_interval_sec)
        self.pace_sec = 0.0  # seconds to pause after each frame so the animation is watchable
        self._last_emit = 0.0

        with open(self.path, "wb") as fh:
            fh.truncate(self.total_bytes)
        self._mm = np.memmap(self.path, dtype=np.uint8, mode="r+", shape=(self.total_bytes,))
        self._write_header(0, 0, 0, 1.0, 0.0)
        self._mm.flush()

    def _write_header(self, seq: int, phase: int, step: int, residual: float, max_val: float) -> None:
        nx, ny, nz = self.shape
        hdr = self._struct.pack(
            "<8I2f16f6I",
            0x5441504D,  # "MPAT"
            1,
            int(seq),
            int(phase),
            int(step),
            int(nx),
            int(ny),
            int(nz),
            float(residual),
            float(max_val),
            *self.affine_f32.tolist(),
            0, 0, 0, 0, 0, 0,
        )
        self._mm[: self.HEADER_BYTES] = np.frombuffer(hdr, dtype=np.uint8)

    def emit_transient_values(
        self,
        vals_t: np.ndarray,
        phase: int,
        step: int,
        residual: float,
        force: bool = False,
    ) -> None:
        now = time.perf_counter()
        if not force and (now - self._last_emit) < self.min_interval_sec:
            return
        self._last_emit = now
        next_seq = self.seq + 1
        slot_idx = next_seq % 2
        offset = self.HEADER_BYTES + slot_idx * self.slot_bytes
        slot_vol = np.ndarray(
            self.shape,
            dtype=np.float32,
            buffer=self._mm,
            offset=offset,
            order="F",
        )
        slot_vol.fill(0.0)
        v_clean = np.clip(np.nan_to_num(np.asarray(vals_t, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
        slot_vol[self.t_ijk] = v_clean
        max_val = float(v_clean.max()) if v_clean.size else 0.0
        self.seq = next_seq
        self._write_header(self.seq, phase, step, residual, max_val)
        self._mm.flush()
        print(
            f"STREAM_FRAME {self.seq} {int(phase)} {int(step)} {float(residual):.6e} {max_val:.6e}",
            flush=True,
        )



def _save_nifti(vol: np.ndarray, affine: np.ndarray, path: Path) -> str:
    arr = np.nan_to_num(np.asarray(vol, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    img = nib.Nifti1Image(arr, np.asarray(affine, dtype=np.float64))
    img.header.set_qform(affine, code=1)
    img.header.set_sform(affine, code=1)
    nib.save(img, str(path))
    return str(path)


def _resample_mask_to_grid(mask_img: nib.Nifti1Image, shape: Tuple[int, int, int], affine: np.ndarray) -> np.ndarray:
    data = np.asarray(mask_img.dataobj) > 0
    if data.ndim > 3:
        data = data[..., 0]
    if data.shape == tuple(shape) and np.allclose(mask_img.affine, affine, atol=1e-3):
        return data.astype(bool)
    # Transform target grid coordinates (i, j, k) -> RAS -> source voxel coordinates
    inv_src = np.linalg.inv(mask_img.affine)
    M = inv_src @ affine
    resampled = ndi.affine_transform(
        data.astype(np.uint8),
        M[:3, :3],
        offset=M[:3, 3],
        output_shape=tuple(shape),
        order=0,
        mode="constant",
        cval=0,
    )
    return resampled > 0


_STAGED_SUBJECT_KEYS = {"sh", "affine", "zooms", "transient", "absorb"}


def _is_staged_subject_npz(path) -> bool:
    """True for an .npz holding a staged subject (fODF SH plus tissue classes)."""
    try:
        with np.load(path) as z:
            return _STAGED_SUBJECT_KEYS.issubset(z.files)
    except Exception:  # noqa: BLE001 - unreadable or not an npz: not a subject bundle
        return False


def _discover_subject_inputs(
    working_dir: Optional[str],
    odx_path: Optional[str],
    wm_threshold: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float], np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Resolve ``(sh, affine, zooms, transient, absorb_class, dseg, wmgm_mask)`` from
    either a staged ``.npz`` bundle, a pyAFQ/MPAT directory, or an ``.odx``/``.fib.gz``/``.nii.gz`` file.
    ``transient``, ``absorb_class`` and ``dseg`` are None when no tissue maps were found.
    """
    from .afq import _on_grid

    # Case 1: working_dir or odx_path points directly to a staged .npz subject file
    for cand in (working_dir, odx_path):
        if cand and Path(cand).is_file() and str(cand).endswith(".npz") and _is_staged_subject_npz(cand):
            z = np.load(cand)
            sh = np.asarray(z["sh"], dtype=np.float32)
            affine = np.asarray(z["affine"], dtype=np.float64)
            zooms = tuple(float(x) for x in z["zooms"][:3])
            transient = np.asarray(z["transient"], dtype=bool)
            absorb = np.asarray(z["absorb"], dtype=np.int8)
            dseg = (
                np.asarray(z["dseg"], dtype=np.uint8)
                if "dseg" in z.files
                else np.where(transient, 1, np.where(absorb > 0, absorb + 1, 0)).astype(np.uint8)
            )
            wmgm = None
            if "seeds" in z.files:
                wmgm = np.zeros(transient.shape, dtype=bool)
                seeds = np.asarray(z["seeds"], dtype=np.int64)
                if seeds.ndim == 2 and seeds.shape[1] == 3 and len(seeds) > 0:
                    wmgm[tuple(seeds.T)] = True
            return sh, affine, zooms, transient, absorb, dseg, wmgm

    # Case 2: working_dir is a directory (pyAFQ derivatives or folder with NIfTIs / .npz)
    sh_img = None
    pve_img = None
    dseg_img = None
    bm_img = None
    wmgm_img = None

    if working_dir and Path(working_dir).is_dir():
        wdir = Path(working_dir)
        # A staged subject .npz holds sh/transient/absorb arrays; other .npz files in the folder
        # (chain caches, transition caches) are not inputs.
        for npz_path in sorted(wdir.glob("*.npz")):
            if _is_staged_subject_npz(npz_path):
                return _discover_subject_inputs(str(npz_path), None, wm_threshold=wm_threshold)

        all_niis = sorted(wdir.rglob("*.nii*"))
        for p in all_niis:
            name = p.name.lower()
            if sh_img is None and any(
                tok in name for tok in ("model-csd", "model-msmt", "model-opdt", "model-rumba", "_fod", "_sh")
            ) and not any(tok in name for tok in ("_fa", "_md", "_cfa", "visitodf")):
                try:
                    img = nib.load(str(p))
                    if img.ndim == 4 and img.shape[-1] in (15, 28, 45, 66):
                        sh_img = img
                except Exception:
                    pass
            elif pve_img is None and "pve" in name:
                try:
                    img = nib.load(str(p))
                    if img.ndim == 4 and img.shape[-1] == 3:
                        pve_img = img
                except Exception:
                    pass
            elif dseg_img is None and "desc-mpat_dseg" in name:
                try:
                    dseg_img = nib.load(str(p))
                except Exception:
                    pass
            elif bm_img is None and "brain_mask" in name:
                try:
                    bm_img = nib.load(str(p))
                except Exception:
                    pass
            elif wmgm_img is None and "wm_gm_interface" in name:
                try:
                    wmgm_img = nib.load(str(p))
                except Exception:
                    pass

    # Case 3: odx_path (or working_dir as a single file) is an .odx, .fib.gz, .fz, or 4D SH .nii.gz
    target_file = odx_path if (odx_path and Path(odx_path).exists()) else (
        working_dir if (working_dir and Path(working_dir).is_file()) else None
    )
    if sh_img is None and target_file is not None:
        tf = str(target_file)
        if tf.endswith((".nii", ".nii.gz")):
            sh_img = nib.load(tf)
        else:
            import mittens
            from mittens.io import default_sphere, sh_basis_matrix

            odf_data = mittens.load_odf_data(tf)
            dims = tuple(int(d) for d in odf_data.dims)
            affine = np.asarray(odf_data.affine, dtype=np.float64)
            zooms = tuple(float(x) for x in np.linalg.norm(affine[:3, :3], axis=0))
            if odf_data.kind == "sh":
                n_coef = odf_data.odfs.shape[1]
                # Ensure descoteaux07 symmetric lmax=8 (45 coeffs) for the MPAT Chain
                sph = default_sphere()
                B_src = sh_basis_matrix(
                    sph,
                    int(odf_data.sh_order or 8),
                    str(odf_data.sh_basis or "descoteaux07"),
                    bool(odf_data.full_basis),
                )
                B_dst = sh_basis_matrix(sph, 8, "descoteaux07", False)
                if odf_data.sh_basis == "descoteaux07" and not odf_data.full_basis and n_coef == 45:
                    sh_rows = np.asarray(odf_data.odfs, dtype=np.float32)
                else:
                    proj = np.linalg.pinv(B_dst) @ B_src
                    sh_rows = (np.asarray(odf_data.odfs, dtype=np.float64) @ proj.T).astype(np.float32)
            else:
                # Sampled amplitudes on sphere -> project to descoteaux07 lmax=8 (45 coeffs)
                verts = np.asarray(odf_data.vertices, dtype=np.float64)
                amps = np.asarray(odf_data.odfs, dtype=np.float64)
                if odf_data.hemisphere and amps.shape[1] * 2 == len(verts):
                    amps = np.hstack([amps, amps])
                B_dst = sh_basis_matrix(verts[: amps.shape[1]], 8, "descoteaux07", False)
                pinv = np.linalg.pinv(B_dst)
                sh_rows = (amps @ pinv.T).astype(np.float32)

            sh = np.zeros(dims + (sh_rows.shape[1],), dtype=np.float32)
            sh[tuple(odf_data.ijk.T)] = sh_rows
            sh_img = nib.Nifti1Image(sh, affine)

    if sh_img is None:
        raise ValueError(
            "Could not find fODF SH NIfTI, .odx, .fib.gz, or .npz in the specified working directory / ODF input."
        )

    sh = np.asarray(sh_img.dataobj, dtype=np.float32)
    affine = np.asarray(sh_img.affine, dtype=np.float64)
    zooms = tuple(float(x) for x in np.linalg.norm(affine[:3, :3], axis=0))
    mask = np.any(sh != 0, axis=-1)
    if bm_img is not None:
        mask &= _on_grid(bm_img, sh_img) > 0.5

    if pve_img is not None:
        csf, gm, wm = np.moveaxis(_on_grid(pve_img, sh_img), -1, 0)
        from skimage.segmentation import find_boundaries

        edge = find_boundaries(mask, mode="inner") & mask
        transient = mask & ~edge & (wm > float(wm_threshold))
        absorb = np.zeros(mask.shape, np.int8)
        absorb[mask & ~transient & (gm >= csf)] = 1
        absorb[mask & ~transient & (gm < csf)] = 2
        absorb[edge] = 3
        cube = np.ones((3, 3, 3), bool)
        lab, n_cc = ndi.label(transient, cube)
        trapped = np.zeros(mask.shape, bool)
        if n_cc:
            ends = ndi.binary_dilation((absorb == 1) | (absorb == 3), cube)
            reach = np.zeros(n_cc + 1, bool)
            reach[np.unique(lab[ends & transient])] = True
            trapped = transient & ~reach[lab]
            if trapped.any():
                transient = transient & ~trapped
                absorb[trapped] = 2
        dseg = np.where(transient, 1, np.where(absorb > 0, absorb + 1, 0)).astype(np.uint8)
        dseg[trapped] = 5
    elif dseg_img is not None:
        dseg = np.rint(_on_grid(dseg_img, sh_img, order=0)).astype(np.uint8)
        if dseg.ndim == 4:
            dseg = dseg[..., 0]
        transient = dseg == 1
        absorb = np.zeros(dseg.shape, dtype=np.int8)
        absorb[dseg == 2] = 1
        absorb[dseg == 3] = 2
        absorb[dseg == 4] = 3
        absorb[dseg == 5] = 2
    else:
        # No tissue maps: the caller has to supply a white-matter mask. Guessing tissue classes
        # from the fODF would make arbitrary voxels (deep white matter included) valid endpoints.
        transient = absorb = dseg = None

    wmgm = None
    if wmgm_img is not None and transient is not None:
        wmgm = (_on_grid(wmgm_img, sh_img, order=0) > 0) & transient

    return sh, affine, zooms, transient, absorb, dseg, wmgm


def _compute_transitions_hash(
    cfg_key: dict,
    max_length_mm: float,
    sh: np.ndarray,
    affine: np.ndarray,
    transient: np.ndarray,
    absorb: np.ndarray,
) -> str:
    """Compute a deterministic 16-hex-char SHA-1 hash of all transition inputs & parameters."""
    h = hashlib.sha1()
    meta = dict(cfg_key)
    meta["max_length_mm"] = round(float(max_length_mm), 3)
    meta["sh_shape"] = list(sh.shape)
    h.update(json.dumps(meta, sort_keys=True).encode())
    h.update(np.ascontiguousarray(affine, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(transient, dtype=np.uint8).tobytes())
    h.update(np.ascontiguousarray(absorb, dtype=np.int8).tobytes())
    # Fast strided sample of c0 ODF values to distinguish subjects with identical grid geometry
    h.update(np.ascontiguousarray(sh[::4, ::4, ::4, 0], dtype=np.float32).tobytes())
    return h.hexdigest()[:16]


def _export_volumes_from_cached_state(
    z_st,
    out_dir: Path,
) -> Tuple[Dict[str, str], Tuple[int, int, int], np.ndarray, int, int, int]:
    """Write the 8 3D NIfTI volumes from a loaded ``mpat_transitions_*.npz`` archive."""
    shape = tuple(int(s) for s in z_st["shape"])
    pshape = tuple(int(s) for s in z_st["pshape"])
    affine = np.asarray(z_st["affine"], dtype=np.float64)
    t_lin = np.asarray(z_st["t_lin"], dtype=np.int64)
    dead_mean = np.asarray(z_st["dead"], dtype=np.float32).reshape(-1, int(z_st["nb"])).mean(axis=1)

    volume_paths: Dict[str, str] = {}
    mit_ijk = np.asarray(z_st["mit_ijk"], dtype=np.int64) if "mit_ijk" in z_st.files else None
    for map_key in (
        "codi_double",
        "codi_single",
        "coasy",
        "p_terminate_double",
        "p_terminate_single",
        "total_amplitude",
        "kl_single",
    ):
        vol = np.zeros(shape, dtype=np.float32)
        arr_key = f"mit_{map_key}"
        if mit_ijk is not None and arr_key in z_st.files:
            vals = np.asarray(z_st[arr_key], dtype=np.float32)
            vol[tuple(mit_ijk.T)] = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        volume_paths[map_key] = _save_nifti(vol, affine, out_dir / f"{map_key}.nii")

    dead_vol = np.zeros(int(np.prod(pshape)), dtype=np.float32)
    dead_vol[t_lin] = dead_mean
    dead_vol = dead_vol.reshape(pshape)[1:-1, 1:-1, 1:-1]
    volume_paths["chain_dead"] = _save_nifti(dead_vol, affine, out_dir / "chain_dead.nii")

    n_wm = int(t_lin.size)
    n_states = int(z_st["dead"].size)
    n_trans = int(z_st["Q_w"].size)
    return volume_paths, shape, affine, n_wm, n_states, n_trans


MAP_KEYS = (
    "codi_double",
    "codi_single",
    "coasy",
    "p_terminate_double",
    "p_terminate_single",
    "total_amplitude",
    "kl_single",
)


def _requested_config(req: Dict[str, Any]):
    """The chain settings a build_transitions request asks for, and its maximum length (mm)."""
    from .chain import ChainConfig

    ptt = str(req.get("preset", "standard")).lower() == "ptt"
    cfg = ChainConfig(
        angle=float(req.get("angle_deg", 35.0)),
        step=float(req.get("step_vox", 1.0)),
        double_odf=bool(req.get("double_odf", True)),
        stop_mass=float(req.get("stop_mass_tau", 1e-4 if ptt else 0.0)),
        fod_thresh=float(req.get("fod_thresh", 0.08 if ptt else 0.05)),
        fod_power=float(req.get("fod_power", 1.5 if ptt else 1.0)),
        curv_sigma=float(req.get("curv_sigma_deg", 18.0 if ptt else 0.0)),
        probe_radius=float(req.get("probe_radius_vox", 0.5 if ptt else 0.0)),
    )
    return cfg, float(req.get("max_length_mm", 250.0))


# build_transitions request keys for each ChainConfig field
_REQUEST_KEYS = {
    "angle": "angle_deg",
    "step": "step_vox",
    "double_odf": "double_odf",
    "stop_mass": "stop_mass_tau",
    "fod_thresh": "fod_thresh",
    "fod_power": "fod_power",
    "curv_sigma": "curv_sigma_deg",
    "probe_radius": "probe_radius_vox",
}


def _check_prebuilt(source, stored_cfg: Dict[str, Any], stored_max_length, req, cfg, max_length_mm) -> None:
    """Prebuilt transitions keep the settings they were built with. Refuse a request that asks for
    other settings or for tissue-mask overrides instead of silently ignoring them. A request that
    names a preset asks for every setting (the preset fills the gaps); otherwise only the settings
    it names are checked."""
    import math
    from dataclasses import asdict

    from .chain import ChainConfig

    valid = {f for f in ChainConfig.__dataclass_fields__}
    built = asdict(ChainConfig(**{k: v for k, v in stored_cfg.items() if k in valid}))
    wanted = asdict(cfg)
    problems = []
    for key, want in wanted.items():
        if key not in _REQUEST_KEYS or ("preset" not in req and _REQUEST_KEYS[key] not in req):
            continue
        have = built[key]
        if isinstance(want, (bool, str)) or isinstance(have, (bool, str)):
            same = have == want
        else:
            same = math.isclose(float(have), float(want), rel_tol=1e-4, abs_tol=1e-6)
        if not same:
            problems.append(f"{key} is {have} in the file but {want} was requested")
    if stored_max_length is not None and "max_length_mm" in req and not math.isclose(
        float(stored_max_length), float(max_length_mm), rel_tol=1e-4
    ):
        problems.append(f"max length is {stored_max_length} mm in the file but {max_length_mm} mm was requested")
    for key, label in (("transient_mask_path", "transient WM mask"), ("absorb_mask_path", "absorbing mask")):
        if req.get(key):
            problems.append(f"a {label} can't be applied to prebuilt transitions")
    if problems:
        raise ValueError(
            f"{Path(source).name} holds prebuilt transitions with fixed settings: "
            + "; ".join(problems)
            + ". Match the node's settings to the file, or point the node at the subject's fODF to "
            "build new transitions."
        )


def _is_transitions_odx(path) -> bool:
    """True for an ODX written by mpat (it carries the chain's stencil arrays), False for a plain fODF ODX."""
    import zipfile

    p = Path(path)
    if p.is_dir():
        return any((p / "groups").glob("mpat_Q_ptr.*"))
    try:
        with zipfile.ZipFile(p) as z:
            return any(n.startswith("groups/mpat_Q_ptr") for n in z.namelist())
    except zipfile.BadZipFile:
        return False


def _mittens_maps(sh_rows: np.ndarray, ijk: np.ndarray, shape, affine: np.ndarray, zooms, cfg):
    """MITTENS transition maps (``MAP_KEYS``) for voxels ``ijk`` with fODF SH rows ``sh_rows``.
    Returns ``(ijk, {key: values})`` for the voxels MITTENS fitted."""
    import mittens
    from mittens.io import OdfData

    n_coef = int(sh_rows.shape[1])
    sym_order = int(round((-3.0 + np.sqrt(1.0 + 8.0 * n_coef)) / 2.0))
    if (sym_order + 1) * (sym_order + 2) // 2 == n_coef:
        sh_order, full_basis = sym_order, False
    else:
        sh_order, full_basis = int(round(np.sqrt(n_coef) - 1.0)), True
    odf_data = OdfData.from_sh(
        sh_rows, ijk, tuple(shape), affine, sh_order=sh_order, sh_basis="descoteaux07", full_basis=full_basis
    )
    step_mm = float(cfg.step * min(zooms))
    sigma_rad = max(np.deg2rad(cfg.curv_sigma), 0.05)
    model = mittens.TransitionModel(
        step_mm=step_mm,
        angle_max=cfg.angle,
        turning="vmf" if cfg.curv_sigma > 0.0 else "flat",
        persistence_length_mm=float(step_mm * sigma_rad**2),
        tau=cfg.stop_mass,
        tau0=cfg.stop_mass,
        odf_power=cfg.fod_power,
    )
    fit = model.fit(odf_data)
    maps = dict(fit.maps())
    maps["kl_single"] = fit.kl_single
    clean = {
        k: np.nan_to_num(np.asarray(maps[k], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0) for k in MAP_KEYS
    }
    return np.asarray(fit.ijk), clean


def cmd_build_transitions(req_path: str) -> None:
    """Execute ``build_transitions`` and emit ``RESULT <json>``."""

    from .chain import Chain
    from .fixels import segment_lobes
    from .solve import overlength_states, prepare_launches, segment_bundles

    t_start = time.perf_counter()
    with open(req_path, "r") as fh:
        req = json.load(fh)

    out_dir = Path(req["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    _emit_progress(1, 6)
    cfg, max_length_mm = _requested_config(req)

    # Fast path 0: Did the user point working_dir or odx_path directly at a precomputed
    # mpat_transitions_*.npz (or *_transitions_dwimap.odx) archive?
    for direct_cand in (req.get("working_dir"), req.get("odx_path")):
        if not direct_cand or not Path(direct_cand).exists():
            continue
        if Path(direct_cand).is_file() and str(direct_cand).endswith(".npz"):
            z_try = np.load(direct_cand)
            if "Q_ptr" in z_try.files and "L_seeds" in z_try.files and "shape" in z_try.files:
                if "cfg_json" not in z_try.files:
                    raise ValueError(f"{Path(direct_cand).name} does not record the settings it was built with")
                _check_prebuilt(direct_cand, json.loads(str(z_try["cfg_json"])), None, req, cfg, max_length_mm)
                _emit_progress(5, 6)
                volume_paths, shape, affine, n_wm, n_states, n_trans = _export_volumes_from_cached_state(
                    z_try, out_dir
                )
                _emit_progress(6, 6)
                elapsed = time.perf_counter() - t_start
                fp_h = str(z_try["fp_hash"]) if "fp_hash" in z_try.files else ""
                result = {
                    "state_path": str(Path(direct_cand).resolve()),
                    "fp_hash": fp_h,
                    "dims": list(shape),
                    "voxel_to_ras": affine.tolist(),
                    "volumes": volume_paths,
                    "summary": (
                        f"Loaded [{Path(direct_cand).name}]: "
                        f"{n_wm:,} WM voxels ({n_states:,} states, {n_trans / 1e6:.1f}M transitions) in {elapsed:.1f}s"
                    ),
                }
                with open(out_dir / "transitions_result.json", "w") as fh:
                    json.dump(result, fh, indent=2)
                print(f"RESULT {json.dumps(result)}", flush=True)
                return
        elif str(direct_cand).endswith((".odx", ".odxd")) and _is_transitions_odx(direct_cand):
            # Transitions written by mpat: reuse the chain as built. A plain fODF ODX falls through
            # to the full build below.
            from .transitions import load_transitions_odx

            name = Path(direct_cand).name
            chain_odx, extras = load_transitions_odx(direct_cand, return_extras=True)
            if "launches" not in extras:
                raise ValueError(f"{name} has no seed launches; rewrite it with mpat run --save-transitions")
            if extras.get("sh_rows") is None:
                raise ValueError(f"{name} has no fODF SH coefficients to compute the transition maps from")
            mpat_meta = extras["header"].get("mpat") or {}
            _check_prebuilt(
                direct_cand, mpat_meta.get("config") or {}, mpat_meta.get("max_length_mm"), req, cfg, max_length_mm
            )
            _emit_progress(4, 6)
            affine = extras["affine"]
            shape = chain_odx.shape
            # The same MITTENS maps the full build computes, from the fODF stored in the ODX
            # (white-matter voxels only: that is all the ODX keeps).
            wm_ijk = np.stack(np.unravel_index(chain_odx.t_lin, chain_odx.pshape), 1) - 1
            map_ijk, maps = _mittens_maps(extras["sh_rows"], wm_ijk, shape, affine, extras["zooms"], chain_odx.cfg)
            volume_paths = {}
            for map_key, vals in maps.items():
                vol = np.zeros(shape, dtype=np.float32)
                vol[tuple(map_ijk.T)] = vals
                volume_paths[map_key] = _save_nifti(vol, affine, out_dir / f"{map_key}.nii")
            dpv = extras.get("dpv") or {}
            dead_mean = dpv.get("dead_mean")
            if dead_mean is None:
                dead_mean = chain_odx.dead.reshape(-1, chain_odx.nb).mean(axis=1)
            volume_paths["chain_dead"] = _save_nifti(
                chain_odx.to_volume(np.asarray(dead_mean, dtype=np.float32).ravel()),
                affine,
                out_dir / "chain_dead.nii",
            )
            _emit_progress(6, 6)
            elapsed = time.perf_counter() - t_start
            result = {
                "state_path": str(Path(direct_cand).resolve()),
                "fp_hash": str(mpat_meta.get("fingerprint", "")),
                "dims": list(shape),
                "voxel_to_ras": affine.tolist(),
                "volumes": volume_paths,
                "summary": (
                    f"Loaded ODX [{name}]: {chain_odx.t_lin.size:,} WM voxels ({chain_odx.n_states:,} states, "
                    f"{chain_odx.n_transitions / 1e6:.1f}M transitions) in {elapsed:.1f}s"
                ),
            }
            with open(out_dir / "transitions_result.json", "w") as fh:
                json.dump(result, fh, indent=2)
            print(f"RESULT {json.dumps(result)}", flush=True)
            return

    wm_threshold = float(req.get("wm_threshold", 0.4))
    sh, affine, zooms, transient, absorb, dseg, wmgm = _discover_subject_inputs(
        req.get("working_dir"),
        req.get("odx_path"),
        wm_threshold=wm_threshold,
    )
    shape = tuple(int(d) for d in sh.shape[:3])
    fodf_mask = np.any(sh != 0, axis=-1)

    # Optional user-wired transient (white-matter) mask
    t_override = None
    if req.get("transient_mask_path") and Path(req["transient_mask_path"]).exists():
        t_override = _resample_mask_to_grid(nib.load(req["transient_mask_path"]), shape, affine)
        if not t_override.any():
            t_override = None
    if transient is None:
        if t_override is None:
            raise ValueError(
                "No tissue maps were found for this fODF (pyAFQ pve_internal, desc-mpat_dseg, or a staged "
                "subject .npz). Wire a Transient WM Mask into the node, or point it at the subject's "
                "pyAFQ folder."
            )
        edge = fodf_mask & ~ndi.binary_erosion(fodf_mask)
        transient = t_override & fodf_mask & ~edge
        absorb = np.zeros(shape, dtype=np.int8)
        absorb[fodf_mask & ~transient] = 1
        absorb[edge] = 3
    elif t_override is not None:
        transient = t_override & fodf_mask
        absorb[transient] = 0
        absorb[(absorb == 0) & fodf_mask & ~transient] = 1

    if req.get("absorb_mask_path") and Path(req["absorb_mask_path"]).exists():
        a_override = _resample_mask_to_grid(nib.load(req["absorb_mask_path"]), shape, affine)
        if a_override.any():
            transient = transient & ~a_override
            absorb[a_override] = 1

    _emit_progress(2, 6)

    fp_hash = _compute_transitions_hash(cfg.key(), max_length_mm, sh, affine, transient, absorb)
    hashed_filename = f"mpat_transitions_{fp_hash}.npz"

    # Determine where to look for / save the hashed transitions file:
    # Prefer the subject's working_dir (if a directory) or parent of odx_path, falling back to cache_dir.
    fallback_cache_dir = Path(req.get("cache_dir") or out_dir.parent)
    fallback_cache_dir.mkdir(parents=True, exist_ok=True)

    candidate_dirs = []
    if req.get("working_dir"):
        wp = Path(req["working_dir"])
        if wp.is_dir():
            candidate_dirs.append(wp)
        elif wp.parent.is_dir():
            candidate_dirs.append(wp.parent)
    if req.get("odx_path"):
        op = Path(req["odx_path"])
        if op.parent.is_dir():
            candidate_dirs.append(op.parent)
    candidate_dirs.append(fallback_cache_dir)

    state_cache_path = None
    for cdir in candidate_dirs:
        p = cdir / hashed_filename
        if p.exists():
            state_cache_path = p
            break
    if state_cache_path is None:
        # Pick the first writable directory (preferring subject working_dir so the hashed .npz stays with the dataset)
        for cdir in candidate_dirs:
            if os.access(cdir, os.W_OK):
                state_cache_path = cdir / hashed_filename
                break
        if state_cache_path is None:
            state_cache_path = fallback_cache_dir / hashed_filename

    # Fast path 1: Hashed transitions archive already exists on disk!
    if state_cache_path.exists():
        _emit_progress(5, 6)
        z_st = np.load(state_cache_path)
        if "mit_ijk" in z_st.files and "Q_ptr" in z_st.files:
            volume_paths, _, _, n_wm, n_states, n_trans = _export_volumes_from_cached_state(z_st, out_dir)
            _emit_progress(6, 6)
            elapsed = time.perf_counter() - t_start
            result = {
                "state_path": str(state_cache_path),
                "fp_hash": fp_hash,
                "dims": list(shape),
                "voxel_to_ras": affine.tolist(),
                "volumes": volume_paths,
                "summary": (
                    f"Cached [{state_cache_path.name}]: {n_wm:,} WM voxels "
                    f"({n_states:,} states, {n_trans / 1e6:.1f}M transitions) in {elapsed:.1f}s"
                ),
            }
            with open(out_dir / "transitions_result.json", "w") as fh:
                json.dump(result, fh, indent=2)
            print(f"RESULT {json.dumps(result)}", flush=True)
            return

    # Full build path:
    # 1. MITTENS v2 transition probability maps across the fODF voxels that walkers use
    odf_mask = (transient | (absorb == 1) | (absorb == 3)) & fodf_mask
    ijk_odf = np.argwhere(odf_mask)
    map_ijk, maps = _mittens_maps(sh[tuple(ijk_odf.T)], ijk_odf, shape, affine, zooms, cfg)

    volume_paths: Dict[str, str] = {}
    mit_arrays: Dict[str, np.ndarray] = {"mit_ijk": map_ijk.astype(np.int32)}
    for map_key, vals in maps.items():
        mit_arrays[f"mit_{map_key}"] = vals
        vol = np.zeros(shape, dtype=np.float32)
        vol[tuple(map_ijk.T)] = vals
        volume_paths[map_key] = _save_nifti(vol, affine, out_dir / f"{map_key}.nii")

    _emit_progress(3, 6)

    # 2. Build or load the 26-class MPAT absorbing Markov chain
    chain_cache_path = state_cache_path.parent / f"mpat_chain_{fp_hash}.npz"

    chain = Chain.build(sh, zooms, transient, absorb, cfg, cache=str(chain_cache_path))
    _emit_progress(4, 6)

    if max_length_mm > 0:
        kill, _, _ = overlength_states(chain, max_length_mm, zooms)
        chain.kill = kill if kill.any() else None
    else:
        chain.kill = None

    # Prepare launches for ALL transient WM voxels so any downstream custom seed_mask works in O(1)
    all_wm_seeds = np.argwhere(transient)
    launches = prepare_launches(chain, all_wm_seeds)
    chain.release_pmfs()

    # Default seed mask is wm_gm_interface if available, else all transient WM
    if wmgm is not None and wmgm.any():
        default_seed_ok = wmgm[tuple(all_wm_seeds.T)]
    else:
        default_seed_ok = np.ones(len(all_wm_seeds), dtype=bool)

    _emit_progress(5, 6)
    t_ijk = np.stack(np.unravel_index(chain.t_lin, chain.pshape), 1) - 1
    sh_rows = np.ascontiguousarray(sh[tuple(t_ijk.T)], dtype=np.float32)
    lobes = segment_lobes(sh_rows, chain.geo)

    # Solve unconditioned whole-brain density using default seeds for AFD attribution
    launches_def = replace(
        launches,
        pair_w=launches.pair_w * default_seed_ok[launches.pair_seed].astype(np.float64),
    )
    whole = segment_bundles(
        chain,
        [{"name": "whole_brain"}],
        launches_def,
        tol=1e-5,
        accel="mr",
    )["whole_brain"]

    dead_mean = chain.dead.reshape(-1, chain.nb).mean(axis=1).astype(np.float32)
    dead_vol = chain.to_volume(dead_mean)
    volume_paths["chain_dead"] = _save_nifti(dead_vol, affine, out_dir / "chain_dead.nii")

    np.savez(
        state_cache_path,
        fp_hash=np.array(fp_hash),
        cfg_json=np.array(json.dumps(cfg.key())),
        shape=np.array(shape, dtype=np.int64),
        pshape=np.array(chain.pshape, dtype=np.int64),
        affine=affine,
        zooms=np.array(zooms, dtype=np.float64),
        label=chain.label,
        t_lin=chain.t_lin,
        tindex=chain.tindex,
        a_lin=chain.a_lin,
        aindex=chain.aindex,
        a_class=chain.a_class,
        noff=chain.noff,
        nb=np.array(chain.nb, dtype=np.int64),
        Q_ptr=chain.Q_ptr,
        Q_slot=chain.Q_slot,
        Q_w=chain.Q_w,
        R_ptr=chain.R_ptr,
        R_ind=chain.R_ind,
        R_val=chain.R_val,
        dead=chain.dead,
        tmpl_n=chain.tmpl_n,
        tmpl_b=chain.tmpl_b,
        kill=chain.kill if chain.kill is not None else np.zeros(0, dtype=bool),
        vbin=chain.geo.vbin,
        # Launches fields
        L_seeds=launches.seeds,
        L_seed_lin=launches.seed_lin,
        L_pair_seed=launches.pair_seed,
        L_pair_w=launches.pair_w,
        L_pair_d=launches.pair_d,
        L_hseed=launches.hseed,
        L_lptr=launches.lptr,
        L_lkey=launches.lkey,
        L_lval=launches.lval,
        L_dead=launches.dead,
        default_seed_ok=default_seed_ok,
        # Fixel lobes & whole-brain reference
        sh_rows=sh_rows,
        lobes_frow=lobes._frow,
        lobes_w_fix=lobes._w_fix,
        lobes_w_unass=lobes._w_unass,
        lobes_dir=lobes.direction,
        lobes_peak=lobes.peak,
        lobes_integral=lobes.integral,
        whole_dc=whole.density_class.astype(np.float32),
        # MITTENS 3D scalar maps
        **mit_arrays,
    )
    n_states = int(chain.n_states)
    n_trans = int(chain.n_transitions)

    _emit_progress(6, 6)
    elapsed = time.perf_counter() - t_start
    result = {
        "state_path": str(state_cache_path),
        "fp_hash": fp_hash,
        "dims": list(shape),
        "voxel_to_ras": affine.tolist(),
        "volumes": volume_paths,
        "summary": (
            f"Saved [{hashed_filename}]: {int(transient.sum()):,} WM voxels "
            f"({n_states:,} states, {n_trans / 1e6:.1f}M transitions) in {elapsed:.1f}s"
        ),
    }
    res_path = out_dir / "transitions_result.json"
    with open(res_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"RESULT {json.dumps(result)}", flush=True)


def _reconstruct_state(state_path: str):
    """Fast O(1) reconstruction of ``(chain, launches, default_seed_ok, sh_rows, lobes, whole_dc, affine, zooms)``
    from ``mpat_transitions_*.npz`` or ``*_transitions_dwimap.odx`` without rebuilding geometry or ODF integrals.
    """
    from .chain import Chain, ChainConfig
    from .fixels import Lobes, _sh_fingerprint, segment_lobes
    from .solve import Launches

    if str(state_path).endswith((".odx", ".odxd")):
        from .transitions import load_transitions_odx

        chain, extras = load_transitions_odx(state_path, return_extras=True)
        affine = extras["affine"]
        zooms = extras["zooms"]
        launches = extras["launches"]
        default_seed_ok = np.ones(len(launches.seeds), dtype=bool)
        sh_rows = extras.get("sh_rows")
        if sh_rows is None:
            sh_rows = np.zeros((chain.t_lin.size, 45), dtype=np.float32)
        lobes = extras.get("lobes")
        if lobes is None:
            lobes = segment_lobes(sh_rows, chain.geo)
        else:
            lobes._sh_fp = _sh_fingerprint(sh_rows)
        whole_dc = extras.get("whole_dc")
        if whole_dc is None:
            whole_dc = np.ones((chain.t_lin.size, chain.nb), dtype=np.float32)
        return chain, launches, default_seed_ok, sh_rows, lobes, whole_dc, affine, zooms

    z = np.load(state_path)
    cfg_dict = json.loads(str(z["cfg_json"]))
    cfg = ChainConfig(**cfg_dict)
    shape = tuple(int(x) for x in z["shape"])
    pshape = tuple(int(x) for x in z["pshape"])
    affine = np.asarray(z["affine"], dtype=np.float64)
    zooms = tuple(float(x) for x in z["zooms"])

    chain = Chain()
    chain.cfg = cfg
    chain.shape = shape
    chain.pshape = pshape
    chain.label = z["label"]
    chain.t_lin = z["t_lin"]
    chain.tindex = z["tindex"]
    chain.a_lin = z["a_lin"]
    chain.aindex = z["aindex"]
    chain.a_class = z["a_class"]
    chain.noff = z["noff"]
    chain.nb = int(z["nb"])
    chain.Q_ptr = z["Q_ptr"]
    chain.Q_slot = z["Q_slot"]
    chain.Q_w = z["Q_w"]
    chain.R_ptr = z["R_ptr"]
    chain.R_ind = z["R_ind"]
    chain.R_val = z["R_val"]
    chain.dead = z["dead"]
    chain.tmpl_n = z["tmpl_n"]
    chain.tmpl_b = z["tmpl_b"]
    kill_arr = z["kill"]
    chain.kill = kill_arr if kill_arr.size > 0 else None
    chain._finalize()

    # Lightweight geometry proxy holding vbin (and full geo only if needed)
    class _GeoProxy:
        pass

    geo = _GeoProxy()
    geo.nb = chain.nb
    geo.vbin = z["vbin"]
    chain.geo = geo

    launches = Launches(
        seeds=z["L_seeds"],
        seed_lin=z["L_seed_lin"],
        pair_seed=z["L_pair_seed"],
        pair_w=z["L_pair_w"],
        pair_d=z["L_pair_d"],
        hseed=z["L_hseed"],
        lptr=z["L_lptr"],
        lkey=z["L_lkey"],
        lval=z["L_lval"],
        dead=z["L_dead"],
    )
    default_seed_ok = np.asarray(z["default_seed_ok"], dtype=bool)
    sh_rows = z["sh_rows"]
    frow = z["lobes_frow"]
    n_vox = len(sh_rows)
    counts = (
        np.bincount(frow, minlength=n_vox).astype(np.int64)
        if frow.size
        else np.zeros(n_vox, dtype=np.int64)
    )
    offset = (
        np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
        if counts.size
        else counts
    )
    w_unass = (
        z["lobes_w_unass"]
        if "lobes_w_unass" in z.files
        else np.zeros((n_vox, chain.nb), dtype=np.float64)
    )
    lobes = Lobes(
        count=counts,
        offset=offset,
        direction=z["lobes_dir"],
        peak=z["lobes_peak"],
        integral=z["lobes_integral"],
        peak_min=0.1,
        merge_ratio=0.8,
        _frow=frow,
        _w_fix=z["lobes_w_fix"],
        _w_unass=w_unass,
        _sh_fp=_sh_fingerprint(sh_rows),
    )
    whole_dc = z["whole_dc"]
    return chain, launches, default_seed_ok, sh_rows, lobes, whole_dc, affine, zooms


def _stream_backward(streamer, chain, it, res, total, term, colsign, n_kinds, log_tol) -> None:
    """Stream a backward-solve frame: the completion wavefront blended with the
    waypoint-conditioned completion probability so far."""
    if streamer is None or not (it == 1 or it % 2 == 0):
        return
    nc, nb, Nt = chain.ncomp, chain.nb, chain.t_lin.size
    w_step = np.clip(term[::nc, :n_kinds].sum(1).reshape(Nt, nb).mean(1), 0.0, None)
    w_max = float(w_step.max())
    if w_max > 1e-20:
        w_step = w_step / w_max
    b_cond = np.clip((total[::nc] @ colsign).reshape(Nt, nb).mean(1), 0.0, None)
    b_max = float(b_cond.max())
    b_vis = 0.60 * (b_cond / b_max) + 0.40 * w_step if b_max > 1e-20 else w_step
    streamer.emit_transient_values(b_vis, phase=1, step=it, residual=res, force=True)
    frac = float(np.clip(np.log10(max(res, 1e-12)) / log_tol, 0.0, 1.0)) if log_tol < 0 else 0.5
    _emit_progress(int(round(frac * 100)), 200)
    if streamer.pace_sec > 0.0:
        time.sleep(streamer.pace_sec)


def _stream_forward(streamer, chain, it, res, total, term, beta, colsign, log_tol) -> None:
    """Stream a forward-solve frame: the bundle density so far blended with this step's wavefront."""
    if streamer is None or not (it == 1 or it % 2 == 0):
        return
    from . import kernel as K

    nc, nb, Nt, Ns = chain.ncomp, chain.nb, chain.t_lin.size, chain.n_states
    beta_c = beta if beta.flags.c_contiguous else np.ascontiguousarray(beta)

    def density(x):
        x = x if x.flags.c_contiguous else np.ascontiguousarray(x)
        return np.clip(K.paired_density(x, beta_c, colsign, Ns, nc).reshape(Nt, nb).sum(1), 0.0, None)

    d_tot, d_wave = density(total), density(term)
    t_max, w_max = float(d_tot.max()), float(d_wave.max())
    if t_max > 1e-20 and w_max > 1e-20:
        d_vis = (0.70 * (d_tot / t_max) + 0.30 * (d_wave / w_max)) * t_max
    else:
        d_vis = d_tot
    streamer.emit_transient_values(d_vis, phase=2, step=it, residual=res, force=True)
    frac = float(np.clip(np.log10(max(res, 1e-12)) / log_tol, 0.0, 1.0)) if log_tol < 0 else 0.5
    _emit_progress(100 + int(round(frac * 95)), 200)
    if streamer.pace_sec > 0.0:
        time.sleep(streamer.pace_sec)


def _ends_outside_absorbers(chain, spec: Dict[str, Any]) -> bool:
    """True when a start/end ROI touches no valid absorbing voxel (an ROI inside white matter).
    Only the custom-region solver can end walks there."""
    from .chain import VALID_ENDS

    a_valid = np.isin(chain.a_class, np.asarray(VALID_ENDS))
    for key in ("start", "end"):
        m = spec.get(key)
        if m is not None and np.any(m):
            padded = np.zeros(chain.pshape, dtype=bool)
            padded[1:-1, 1:-1, 1:-1] = m
            if not np.any(padded.ravel()[chain.a_lin] & a_valid):
                return True
    return False


def _solve_bundle_with_custom_regions(
    chain,
    launches,
    spec: Dict[str, Any],
    term_mask: Optional[np.ndarray],
    no_end_mask: Optional[np.ndarray],
    tol: float = 1e-6,
    accel: Any = "mr",
    streamer: Optional[LiveDensityStreamer] = None,
):
    """Solve a bundle whose regions mpat.solve can't express: absorption inside white matter
    (``term_mask``, or start/end ROIs that touch no valid absorbing voxel) and ``no_end_mask``.
    Bundles without them go through :func:`mpat.solve.segment_bundles`, as in ``mpat run``.
    Supports:
    - ``include`` (waypoint ROIs),
    - ``exclude`` (avoid ROIs + outside of limiting_mask),
    - ``start`` / ``end`` (endpoint ROIs),
    - ``term_mask`` (arbitrary absorption ROIs — including interior white-matter voxels!),
    - ``no_end_mask`` (forbidden absorption ROIs where walkers may not terminate).
    """
    from . import kernel as K
    from .chain import VALID_ENDS
    from .solve import BundleResult, _endpoint_kinds, _Plan, _solve

    has_custom_term = term_mask is not None and np.any(term_mask)
    has_no_end = no_end_mask is not None and np.any(no_end_mask)

    # Check if start or end mask is placed inside interior transient WM with no boundary overlap
    pad = np.zeros(chain.pshape, dtype=bool)

    def _padded(m: np.ndarray) -> np.ndarray:
        p = pad.copy()
        p[1:-1, 1:-1, 1:-1] = m
        return p.ravel()

    a_valid = np.isin(chain.a_class, np.asarray(VALID_ENDS))
    interior_absorb = np.zeros(chain.shape, dtype=bool)
    if has_custom_term:
        interior_absorb |= np.asarray(term_mask, dtype=bool)
        a_valid = a_valid | _padded(term_mask)[chain.a_lin]

    for k in ("start", "end"):
        m = spec.get(k)
        if m is not None and np.any(m):
            m_pad = _padded(m)
            # If a user-supplied start/end ROI has no valid boundary absorbing voxels (e.g. a 3D sphere
            # placed inside subcortical white matter), make its interior voxels absorbing automatically!
            if not np.any(m_pad[chain.a_lin] & a_valid):
                interior_absorb |= np.asarray(m, dtype=bool)
                a_valid = a_valid | m_pad[chain.a_lin]

    # General exact solver supporting arbitrary interior absorption (term_mask), no_end_mask, and N waypoints
    p = _Plan(chain, spec, VALID_ENDS)
    nb, Ns, nc = chain.nb, chain.n_states, chain.ncomp

    # Recompute endpoint kinds if a_valid or no_end_mask or term_mask modified valid endpoints
    ends = [_padded(spec[k])[chain.a_lin] if spec.get(k) is not None else None for k in ("start", "end")]
    if has_custom_term:
        t_a = _padded(term_mask)[chain.a_lin]
        # Allow absorption in term_mask to satisfy endpoint completion
        if ends[0] is not None and ends[1] is not None:
            ends[1] = ends[1] | t_a
        elif ends[0] is not None:
            ends[0] = ends[0] | t_a
        elif ends[1] is not None:
            ends[1] = ends[1] | t_a

    if has_no_end:
        ne_a = _padded(no_end_mask)[chain.a_lin]
        a_valid = a_valid & ~ne_a

    kinds, p.M, p.end_rule = _endpoint_kinds(a_valid, *ends)
    p.nK = len(kinds)
    p.C = p.nX * p.nK
    p.xcol = np.repeat(np.arange(p.nX), p.nK)
    p.kcol = np.tile(np.arange(p.nK), p.nX)
    p.absw = np.stack(
        [kinds[p.kcol[c]] & ~p.taboo[chain.a_lin, p.xcol[c]] for c in range(p.C)],
        axis=1,
    ).astype(np.float64)

    # Interior absorption on transient voxels (term_t)
    int_pad = _padded(interior_absorb)
    if has_no_end:
        int_pad &= ~_padded(no_end_mask)
    term_t = int_pad[chain.t_lin]
    has_int_abs = bool(np.any(term_t))

    r_rhs = chain.absorb_rhs(p.absw)
    if has_int_abs:
        term_state = np.repeat(term_t, nb)
        # Stop walkers upon reaching term_state (they absorb rather than stepping out)
        p.mask[term_state, :] = 0

        # Determine endpoint kind weights for interior absorbing voxels on chain.t_lin
        ends_t = [_padded(spec[k])[chain.t_lin] if spec.get(k) is not None else None for k in ("start", "end")]
        if has_custom_term:
            t_t = _padded(term_mask)[chain.t_lin]
            if ends_t[0] is not None and ends_t[1] is not None:
                ends_t[1] = ends_t[1] | t_t
            elif ends_t[0] is not None:
                ends_t[0] = ends_t[0] | t_t
            elif ends_t[1] is not None:
                ends_t[1] = ends_t[1] | t_t
        kinds_t, _, _ = _endpoint_kinds(term_t, *ends_t)
        term_absw_t = np.stack(
            [kinds_t[p.kcol[c]] & ~p.taboo[chain.t_lin, p.xcol[c]] for c in range(p.C)],
            axis=1,
        ).astype(np.float64)
        term_absw_s = np.repeat(term_absw_t, nb, axis=0)
        q_into_term = np.empty_like(r_rhs)
        chain.mul_backward(np.ascontiguousarray(term_absw_s), q_into_term)
        r_rhs = r_rhs + q_into_term

    mask = p.mask.copy()
    r_rhs *= mask
    colsign = p.signs[p.xcol]
    log_tol = float(np.log10(max(tol, 1e-12)))

    def _on_bwd_step(it: int, res: float, total: np.ndarray, term: np.ndarray) -> None:
        _stream_backward(streamer, chain, it, res, total, term, colsign, p.nK, log_tol)

    bwd_accel = False if (streamer is not None and getattr(streamer, "pace_sec", 0.0) > 0.0) else accel
    beta, info_b = _solve(
        chain, r_rhs, mask, tol, 5000, "backward", verbose=False, accel=bwd_accel, on_step=_on_bwd_step
    )
    if has_int_abs:
        beta[term_state, :] = term_absw_s[term_state, :]

    P = launches.pair_seed.size
    seed_ok = (~p.taboo[launches.hseed][:, p.xcol]).astype(np.float64)
    vals = np.empty((launches.hseed.size, p.C), dtype=np.float64)
    K.half_values(
        launches.hseed,
        launches.lptr,
        launches.lkey,
        launches.lval,
        chain.label,
        chain.noff,
        chain.tindex,
        chain.aindex,
        nb,
        np.ascontiguousarray(beta),
        p.absw,
        seed_ok,
        vals,
    )
    V = vals.reshape(P, 2, p.nX, p.nK)
    pb = np.einsum("pxk,kl,pxl->px", V[:, 0], p.M, V[:, 1]) @ p.signs
    seed_prob = np.bincount(launches.pair_seed, weights=launches.pair_w * pb, minlength=len(launches.seeds))
    coef = np.einsum("phxl,kl->phxk", V[:, ::-1], p.M).reshape(2 * P, p.C)
    coef *= np.repeat(launches.pair_w, 2)[:, None] * seed_ok
    pi0 = np.zeros((Ns * nc, p.C), dtype=np.float64)
    a0 = np.zeros((chain.a_lin.size, p.C), dtype=np.float64)
    K.scatter_init(
        launches.hseed,
        launches.lptr,
        launches.lkey,
        launches.lval,
        coef,
        chain.label,
        chain.noff,
        chain.tindex,
        chain.aindex,
        nb,
        p.taboo,
        p.xcol,
        pi0,
        a0,
    )
    if has_int_abs:
        a0_term_s = pi0[term_state, :].copy()
        pi0[term_state, :] = 0.0
    if getattr(chain, "kill", None) is not None:
        pi0[np.repeat(chain.kill, nc)] = 0.0

    beta_c = np.ascontiguousarray(beta)

    def _on_fwd_step(it: int, res: float, total: np.ndarray, term: np.ndarray) -> None:
        _stream_forward(streamer, chain, it, res, total, term, beta_c, colsign, log_tol)

    # When live streaming is active, run the forward pass unaccelerated (accel=False)
    # so the Markov chain steps smoothly t=1,2,3,... and animates the bundle growth.
    fwd_accel = False if (streamer is not None and getattr(streamer, "pace_sec", 0.0) > 0.0) else accel
    S, info_f = _solve(
        chain, pi0, mask, tol, 5000, "forward", verbose=False, accel=fwd_accel, on_step=_on_fwd_step
    )
    S_p = np.ascontiguousarray(S)
    dcls = K.paired_density(S_p, beta_c, colsign, Ns, nc).reshape(chain.t_lin.size, nb)

    ti = chain.tindex[launches.seed_lin]
    pt = ti[launches.pair_seed]
    on = pt >= 0
    w0 = 0.5 * (launches.pair_w * pb)[on]
    for k in (0, 1):
        np.add.at(dcls, (pt[on], chain.geo.vbin[launches.pair_d[on, k]]), w0)
    density = dcls.sum(1)
    endpoints = ((a0 + chain.absorbed_from(S_p)) * p.absw) @ colsign

    # Interior absorbed endpoints (mapped onto nearest boundary absorbing voxels or returned as volume)
    interior_endpoints_t = None
    if has_int_abs:
        q_fwd_term = np.empty_like(S_p)
        chain.mul_forward(S_p, q_fwd_term)
        int_ep_s = ((a0_term_s + q_fwd_term[term_state, :]) * term_absw_s[term_state, :]) @ colsign
        interior_endpoints_t = np.zeros(chain.t_lin.size, dtype=np.float64)
        interior_endpoints_t[term_t] = np.clip(int_ep_s.reshape(-1, nb).sum(1), 0.0, None)
        density[term_t] += interior_endpoints_t[term_t]

    f26 = np.empty((chain.t_lin.size, 26), dtype=np.float64)
    K.stencil_flux(
        chain.Q_ptr,
        chain.Q_slot,
        chain.Q_w,
        chain.tmpl_n,
        chain.tmpl_b,
        chain.nbr_t,
        chain.nb,
        S_p,
        np.ascontiguousarray(beta),
        colsign,
        f26,
    )
    flux26 = np.clip(f26, 0.0, None).astype(np.float32)

    info = dict(
        n_seeds=int(len(launches.seeds)),
        n_streamlines=int(P),
        expected_bundle_streamlines=float(np.clip(seed_prob, 0.0, None).sum()),
        endpoint_rule=p.end_rule,
        n_columns=int(p.C),
        tolerance=tol,
        backward=info_b,
        forward=info_f,
    )
    res = BundleResult(
        p.name,
        launches.seeds,
        np.clip(seed_prob, 0.0, None),
        np.clip(density, 0.0, None),
        np.clip(endpoints, 0.0, None),
        info,
        np.clip(dcls, 0.0, None).astype(np.float32),
        flux26=flux26,
        start_mask=p.start_mask,
        end_mask=p.end_mask,
    )
    res.interior_endpoints_t = interior_endpoints_t
    return res


def cmd_solve_bundle(req_path: str) -> None:
    """Execute ``solve_bundle`` and emit ``RESULT <json>``."""
    from .afq import seal_26 as apply_seal_26
    from .branches import extract_branch_arbor, extract_flux_streamlines, write_trx
    from .quantify import quantify_bundle
    from .solve import segment_bundles

    t_start = time.perf_counter()
    with open(req_path, "r") as fh:
        req = json.load(fh)

    out_dir = Path(req["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    _emit_progress(1, 5)
    chain, launches, default_seed_ok, sh_rows, lobes, whole_dc, affine, zooms = _reconstruct_state(
        req["state_path"]
    )
    shape = chain.shape

    # 1. Configure seed mask (either custom seed_mask_path or default_seed_ok)
    if req.get("seed_mask_path") and Path(req["seed_mask_path"]).exists():
        s_mask = _resample_mask_to_grid(nib.load(req["seed_mask_path"]), shape, affine)
        seed_ok = s_mask[tuple(launches.seeds.T)]
    else:
        seed_ok = default_seed_ok
    active_launches = replace(
        launches,
        pair_w=launches.pair_w * seed_ok[launches.pair_seed].astype(np.float64),
    )

    # 2. Load and resample ROIs from TrackingPlan
    do_seal = bool(req.get("seal_26", True))
    include_masks: List[np.ndarray] = []
    for inc_path in req.get("include_mask_paths", []):
        if inc_path and Path(inc_path).exists():
            m = _resample_mask_to_grid(nib.load(inc_path), shape, affine)
            if m.any():
                include_masks.append(apply_seal_26(m) if do_seal else m)

    exclude_mask: Optional[np.ndarray] = None
    if req.get("exclude_mask_path") and Path(req["exclude_mask_path"]).exists():
        m = _resample_mask_to_grid(nib.load(req["exclude_mask_path"]), shape, affine)
        if m.any():
            exclude_mask = apply_seal_26(m) if do_seal else m

    if req.get("limiting_mask_path") and Path(req["limiting_mask_path"]).exists():
        lim = _resample_mask_to_grid(nib.load(req["limiting_mask_path"]), shape, affine)
        if lim.any():
            outside_lim = ~lim
            exclude_mask = outside_lim if exclude_mask is None else (exclude_mask | outside_lim)

    start_mask: Optional[np.ndarray] = None
    if req.get("start_mask_path") and Path(req["start_mask_path"]).exists():
        m = _resample_mask_to_grid(nib.load(req["start_mask_path"]), shape, affine)
        if m.any():
            start_mask = m

    end_mask: Optional[np.ndarray] = None
    if req.get("end_mask_path") and Path(req["end_mask_path"]).exists():
        m = _resample_mask_to_grid(nib.load(req["end_mask_path"]), shape, affine)
        if m.any():
            end_mask = m

    term_mask: Optional[np.ndarray] = None
    if req.get("term_mask_path") and Path(req["term_mask_path"]).exists():
        m = _resample_mask_to_grid(nib.load(req["term_mask_path"]), shape, affine)
        if m.any():
            term_mask = m

    no_end_mask: Optional[np.ndarray] = None
    if req.get("no_end_mask_path") and Path(req["no_end_mask_path"]).exists():
        m = _resample_mask_to_grid(nib.load(req["no_end_mask_path"]), shape, affine)
        if m.any():
            no_end_mask = m

    # Optional midline exclusion: keep a one-sided bundle in its hemisphere. True picks the side
    # the way mpat.run does (the name, "Left ..." / "..._L", else the ROI centres); "left" /
    # "right" force it.
    midline_raw = req.get("midline_exclusion", False)
    midline_mode = str(midline_raw).strip().lower() if midline_raw not in (None, False, True) else midline_raw
    if midline_mode in ("true", "on", "auto"):
        midline_mode = True
    elif midline_mode in ("false", "off", "none", ""):
        midline_mode = False
    midline_tol_mm = float(req.get("midline_tol_mm", 0.0))
    bundle_name = str(req.get("bundle_name") or "Bundle")
    if midline_mode:
        from .afq import _is_left

        xyz_x = nib.affines.apply_affine(affine, np.indices(shape).reshape(3, -1).T)[:, 0].reshape(shape)
        if midline_mode in ("left", "right"):
            is_left = midline_mode == "left"
        else:
            is_left = _is_left(bundle_name, {"include": include_masks, "start": start_mask, "end": end_mask}, xyz_x)
        mid_exc = (xyz_x > midline_tol_mm) if is_left else (xyz_x < -midline_tol_mm)
        exclude_mask = mid_exc if exclude_mask is None else (exclude_mask | mid_exc)

    spec: Dict[str, Any] = {
        "name": bundle_name,
        "include": include_masks,
        "exclude": exclude_mask,
        "start": start_mask,
        "end": end_mask,
    }

    _emit_progress(2, 5)
    tol = float(req.get("solver_tol") or req.get("tol") or 1e-6)
    accel_str = str(req.get("solver_accel") or req.get("accel") or "mr").lower()
    accel: Any = False if accel_str in ("none", "false", "") else accel_str
    explicit_stream = bool(req.get("stream_mmap_path"))
    stream_mmap_path = req.get("stream_mmap_path") or str(out_dir / "live_density.mmap")
    streamer: Optional[LiveDensityStreamer] = None
    try:
        streamer = LiveDensityStreamer(Path(stream_mmap_path), chain, affine)
        streamer.pace_sec = 0.035 if explicit_stream else 0.0
    except Exception:
        streamer = None

    if term_mask is not None or no_end_mask is not None or _ends_outside_absorbers(chain, spec):
        res = _solve_bundle_with_custom_regions(
            chain,
            active_launches,
            spec,
            term_mask=term_mask,
            no_end_mask=no_end_mask,
            tol=tol,
            accel=accel,
            streamer=streamer,
        )
    else:
        # The same solver mpat run uses. Live streaming runs it unaccelerated so the frames
        # advance one step at a time.
        log_tol = float(np.log10(max(tol, 1e-12)))

        def on_step(st) -> None:
            if st.phase == "backward":
                _stream_backward(streamer, chain, st.it, st.res, st.total, st.term, st.colsign, st.n_kinds, log_tol)
            else:
                _stream_forward(streamer, chain, st.it, st.res, st.total, st.term, st.beta, st.colsign, log_tol)

        pacing = streamer is not None and streamer.pace_sec > 0.0
        res = segment_bundles(
            chain,
            [spec],
            active_launches,
            tol=tol,
            accel=False if pacing else accel,
            on_step=on_step if streamer is not None else None,
        )[bundle_name]
        res.interior_endpoints_t = None

    _emit_progress(3, 5)
    qb = quantify_bundle(
        chain,
        res,
        whole_dc,
        sh_rows,
        affine,
        lobes=lobes,
        start_mask=start_mask,
        end_mask=end_mask if end_mask is not None else term_mask,
        compute_boundary_conductance=False,
    )
    if streamer is not None:
        streamer.emit_transient_values(res.density_t, phase=3, step=int(res.info["forward"]["iterations"]), residual=0.0, force=True)

    afd_vol = chain.to_volume(qb.voxel_afd_t)
    density_vol = chain.to_volume(res.density_t)
    tdimm_vol = chain.to_volume(qb.tdi_mm_t)
    qharm_vol = chain.to_volume(qb.q_harmonic_t)
    coh_vol = chain.to_volume(qb.flux_coherence_t)
    keff_vol = chain.to_volume(qb.branching_k_eff_t)
    endpoints_vol = chain.to_volume(res.endpoints_a, "absorbing")
    if getattr(res, "interior_endpoints_t", None) is not None:
        endpoints_vol = endpoints_vol + chain.to_volume(res.interior_endpoints_t, "transient")

    volume_paths = {
        "afd": _save_nifti(afd_vol, affine, out_dir / "afd.nii"),
        "density": _save_nifti(density_vol, affine, out_dir / "density.nii"),
        "tdimm": _save_nifti(tdimm_vol, affine, out_dir / "tdimm.nii"),
        "qharmonic": _save_nifti(qharm_vol, affine, out_dir / "qharmonic.nii"),
        "coherence": _save_nifti(coh_vol, affine, out_dir / "coherence.nii"),
        "keff": _save_nifti(keff_vol, affine, out_dir / "keff.nii"),
        "endpoints": _save_nifti(endpoints_vol, affine, out_dir / "endpoints.nii"),
    }

    _emit_progress(4, 5)
    n_streamlines = int(req.get("n_streamlines", 800))
    afd_mass_fraction = float(req.get("afd_mass_fraction", 0.96))
    max_end_branches = int(req.get("n_arbor_branches", 6))
    max_lateral_offshoots = int(req.get("max_lateral_offshoots", 8))
    min_branch_sep_mm = float(req.get("min_branch_sep_mm", 7.0))
    want_streamlines = bool(req.get("extract_streamlines", True))
    want_arbor = bool(req.get("extract_arbor", True))
    # The RK4 step arrives in voxels; extract_flux_streamlines takes millimetres (it divides by
    # the mean voxel size), and its max_steps bounds each half of the bidirectional integration.
    mean_zoom = float(np.mean(np.sqrt((np.asarray(affine, float)[:3, :3] ** 2).sum(0))))
    step_mm = max(float(req.get("step_vox", 0.5)), 1e-3) * mean_zoom
    min_length_mm = float(req.get("min_length_mm", 18.0))
    max_length_mm = float(req.get("max_length_mm", 250.0))
    max_steps = max(1, int(np.ceil(max_length_mm / (2.0 * step_mm))))

    warnings: List[str] = []

    def _warn(stage: str, err: Exception) -> None:
        import traceback

        traceback.print_exc(file=sys.stderr)
        warnings.append(f"{stage} failed: {type(err).__name__}: {err}")

    trx_path = out_dir / "bundle.trx"
    fg = None
    n_branches = 0
    if float(afd_vol.max()) > 0 and float(res.density_t.max()) > 0:
        arbor = None
        if want_arbor:
            try:
                arbor = extract_branch_arbor(
                    afd_vol,
                    endpoints_vol,
                    affine,
                    q_vol=qharm_vol,
                    coherence_vol=coh_vol,
                    keff_vol=keff_vol,
                    start_mask=start_mask,
                    end_mask=end_mask if end_mask is not None else term_mask,
                    bundle_name=bundle_name,
                    max_end_branches=max_end_branches,
                    max_lateral_offshoots=max_lateral_offshoots,
                    min_branch_sep_mm=min_branch_sep_mm,
                )
                n_branches = len(arbor.branches)
            except Exception as err:
                _warn("arbor extraction", err)

        if want_streamlines or arbor is not None:
            try:
                # With extract_streamlines off, n_streamlines=0 keeps only the arbor ridge curves.
                fg = extract_flux_streamlines(
                    chain,
                    res.flux26,
                    afd_vol,
                    affine,
                    q_vol=qharm_vol,
                    coherence_vol=coh_vol,
                    keff_vol=keff_vol,
                    arbor=arbor,
                    n_streamlines=n_streamlines if want_streamlines else 0,
                    step_mm=step_mm,
                    max_steps=max_steps,
                    min_length_mm=min_length_mm,
                    afd_mass_fraction=afd_mass_fraction,
                )
            except Exception as err:
                _warn("flux streamline extraction", err)
    else:
        warnings.append("empty bundle: nothing to extract streamlines from")

    if fg is None:
        from .branches import FluxTractogram

        fg = FluxTractogram(
            streamlines=[],
            dpv={},
            dps={},
            groups={},
            affine=affine,
            dimensions=shape,
        )
    write_trx(fg, str(trx_path))
    n_extracted = len(fg.streamlines)

    _emit_progress(5, 5)
    elapsed = time.perf_counter() - t_start
    exp_sl = float(res.info["expected_bundle_streamlines"])
    afd_vol_mm3 = float(qb.summary.get("tract_afd_volume_mm3", 0.0))
    summary = (
        f"{n_extracted} flux streamlines ({n_branches} branches), "
        f"{exp_sl:.1f} expected walks, {afd_vol_mm3 / 1000.0:.2f} cm³ AFD in {elapsed:.2f}s"
    )
    if warnings:
        summary += " — " + "; ".join(warnings)
    result = {
        "trx_path": str(trx_path),
        "volumes": volume_paths,
        "expected_streamlines": exp_sl,
        "n_flux_streamlines": n_extracted,
        "n_branches": n_branches,
        "quantification": qb.summary,
        "warnings": warnings,
        "summary": summary,
    }
    with open(out_dir / "solve_result.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"RESULT {json.dumps(result)}", flush=True)


def cmd_precompute_cli(argv: List[str]) -> int:
    """CLI helper to precompute ``mpat_transitions_<hash>.npz`` directly from an ODX file or subject folder."""
    import argparse
    import tempfile

    p = argparse.ArgumentParser(
        prog="python -m mpat.trxviz_bridge precompute",
        description="Precompute and cache mpat_transitions_<hash>.npz for instant loading in TRXViz.",
    )
    p.add_argument("input_path", help="Path to an .odx / .nii.gz / .npz file or a pyAFQ/MPAT subject directory")
    p.add_argument("--out-dir", help="Directory to store the hashed mpat_transitions_<hash>.npz (default: alongside input)")
    p.add_argument("--preset", choices=("standard", "ptt"), default="standard")
    p.add_argument("--angle-deg", type=float, default=35.0)
    p.add_argument("--step-vox", type=float, default=1.0)
    p.add_argument("--single-odf", action="store_true", help="Use single-ODF transitions instead of double-ODF")
    p.add_argument("--fod-thresh", type=float, default=None)
    p.add_argument("--fod-power", type=float, default=None)
    p.add_argument("--stop-mass-tau", type=float, default=None)
    p.add_argument("--curv-sigma-deg", type=float, default=None)
    p.add_argument("--probe-radius-vox", type=float, default=None)
    p.add_argument("--max-length-mm", type=float, default=250.0)
    p.add_argument("--wm-threshold", type=float, default=0.4)
    a = p.parse_args(argv)

    inp = Path(a.input_path).resolve()
    target_dir = Path(a.out_dir).resolve() if a.out_dir else (inp if inp.is_dir() else inp.parent)
    target_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="mpat_precompute_") as tmp:
        req = {
            "out_dir": tmp,
            "cache_dir": str(target_dir),
            "working_dir": str(inp) if inp.is_dir() else str(target_dir),
            "odx_path": str(inp) if inp.is_file() else None,
            "preset": a.preset,
            "angle_deg": a.angle_deg,
            "step_vox": a.step_vox,
            "double_odf": not a.single_odf,
            "max_length_mm": a.max_length_mm,
            "wm_threshold": a.wm_threshold,
        }
        # Unset options take the preset's defaults (those of mpat.run) in _requested_config.
        for key, value in (
            ("fod_thresh", a.fod_thresh),
            ("fod_power", a.fod_power),
            ("stop_mass_tau", a.stop_mass_tau),
            ("curv_sigma_deg", a.curv_sigma_deg),
            ("probe_radius_vox", a.probe_radius_vox),
        ):
            if value is not None:
                req[key] = value
        req_file = Path(tmp) / "req.json"
        req_file.write_text(json.dumps(req, indent=2))
        cmd_build_transitions(str(req_file))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in ("build_transitions", "solve_bundle", "precompute"):
        print(
            "Usage:\n"
            "  python -m mpat.trxviz_bridge precompute <input_path> [options]\n"
            "  python -m mpat.trxviz_bridge <build_transitions|solve_bundle> <request.json>",
            file=sys.stderr,
        )
        return 2
    cmd = args[0]
    if cmd == "precompute":
        return cmd_precompute_cli(args[1:])
    if len(args) < 2:
        return 2
    req_path = args[1]
    if cmd == "build_transitions":
        cmd_build_transitions(req_path)
    elif cmd == "solve_bundle":
        cmd_solve_bundle(req_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
