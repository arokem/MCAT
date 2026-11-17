"""Persistence of MPAT and MITTENS transition probabilities in ODX (``.odx``) and NIfTI formats.

* **ODX (``.odx``)** is the native container for 2nd-order / PTT (Parallel Transport Tractography)
  and fixel-aware Markov chains: it stores the masked white-matter voxel grid (``mask.uint8``),
  segmented fODF fixel lobes (``offsets.uint32``, ``directions.3.float32``, ``dpf/*``),
  per-voxel scalar & 26-direction transition summaries (``dpv/*``), and the compressed
  26-class Markov stencil (``groups/mpat_*``) in a single spec-compliant ``.odx`` archive
  that opens directly in ``odx-rs``, ``TRXViz``, and :meth:`mpat.Chain.load_odx`.
* **NIfTI (``.nii.gz`` + ``.json``)** is the classic 1st-order MITTENS representation: a 4-D
  ``(X, Y, Z, 26)`` volume of voxel-to-neighbour transition probabilities plus 3-D scalar
  dispersion (``codi``) and asymmetry (``coasy``) maps and a JSON sidecar listing the 26
  neighbour offsets.
"""
from __future__ import annotations

import json
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import nibabel as nib
import numpy as np
from numba import njit, prange

from . import kernel as K
from .chain import ABSORB_CLASSES, CSF, Chain, ChainConfig
from .fixels import Lobes
from .geometry import OFFSETS, build_geometry
from .solve import Launches

_DTYPE_TO_ODX = {
    np.dtype("float16"): "float16",
    np.dtype("float32"): "float32",
    np.dtype("float64"): "float64",
    np.dtype("int8"): "int8",
    np.dtype("int16"): "int16",
    np.dtype("int32"): "int32",
    np.dtype("int64"): "int64",
    np.dtype("uint8"): "uint8",
    np.dtype("uint16"): "uint16",
    np.dtype("uint32"): "uint32",
    np.dtype("uint64"): "uint64",
}
_ODX_TO_DTYPE = {v: k for k, v in _DTYPE_TO_ODX.items()}


def neighbor_labels(affine: np.ndarray) -> list[str]:
    """Human-readable labels (e.g. ``R_A_S`` or ``i+1_j+0_k-1``) for the 26 lexicographic offsets."""
    try:
        axcodes = nib.orientations.aff2axcodes(np.asarray(affine, dtype=float))
    except Exception:  # noqa: BLE001
        axcodes = ("I", "J", "K")
    opp = {"R": "L", "L": "R", "A": "P", "P": "A", "S": "I", "I": "S"}
    labels = []
    for di, dj, dk in OFFSETS:
        parts = []
        for delta, pos_code in zip((di, dj, dk), axcodes):
            if delta > 0:
                parts.append(pos_code)
            elif delta < 0:
                parts.append(opp.get(pos_code, f"-{pos_code}"))
        labels.append("".join(parts) if parts else "0")
    return labels


@njit(parallel=True, cache=True)
def _marginal_stencil_numba(
    Q_ptr: np.ndarray,
    Q_slot: np.ndarray,
    Q_w: np.ndarray,
    R_ptr: np.ndarray,
    R_ind: np.ndarray,
    R_val: np.ndarray,
    t_lin: np.ndarray,
    a_lin: np.ndarray,
    noff: np.ndarray,
    dead: np.ndarray,
    tmpl_n: np.ndarray,
    nb: int,
    Nt: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse the (Nt * nb) stencil over incoming classes into (Nt, 26) marginal
    voxel-to-neighbour probabilities (including both transient and absorbing neighbours),
    (Nt,) absorption probabilities, and (Nt,) dead probabilities.
    """
    p_nbr = np.zeros((Nt, 26), dtype=np.float32)
    p_abs = np.zeros(Nt, dtype=np.float32)
    p_dead = np.zeros(Nt, dtype=np.float32)
    inv_nb = 1.0 / float(nb)

    for v in prange(Nt):
        v_lin = t_lin[v]
        s0 = v * nb
        acc_abs = 0.0
        acc_dead = 0.0
        for b in range(nb):
            s = s0 + b
            acc_dead += dead[s]
            for k in range(Q_ptr[s], Q_ptr[s + 1]):
                slot = Q_slot[k]
                nbr = tmpl_n[b, slot]
                if nbr >= 0:
                    p_nbr[v, nbr] += np.float32(Q_w[k] * inv_nb)
            for k in range(R_ptr[s], R_ptr[s + 1]):
                w_r = R_val[k]
                acc_abs += w_r
                target_lin = a_lin[R_ind[k]]
                diff = target_lin - v_lin
                for n in range(26):
                    if noff[n] == diff:
                        p_nbr[v, n] += np.float32(w_r * inv_nb)
                        break
        p_abs[v] = np.float32(acc_abs * inv_nb)
        p_dead[v] = np.float32(acc_dead * inv_nb)

    return p_nbr, p_abs, p_dead


@njit(parallel=True, cache=True)
def _fixel_half_probs_numba(
    Q_ptr: np.ndarray,
    Q_slot: np.ndarray,
    Q_w: np.ndarray,
    tmpl_n: np.ndarray,
    nb: int,
    frow: np.ndarray,
    b_plus: np.ndarray,
    b_minus: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (N_fixels, 26) outgoing neighbour transition probabilities along the + and -
    half-lobe direction classes of each fixel.
    """
    nf = frow.size
    p_plus = np.zeros((nf, 26), dtype=np.float32)
    p_minus = np.zeros((nf, 26), dtype=np.float32)
    for f in prange(nf):
        v = frow[f]
        bp = b_plus[f]
        bm = b_minus[f]
        sp = v * nb + bp
        for k in range(Q_ptr[sp], Q_ptr[sp + 1]):
            nbr = tmpl_n[bp, Q_slot[k]]
            if nbr >= 0:
                p_plus[f, nbr] += np.float32(Q_w[k])
        sm = v * nb + bm
        for k in range(Q_ptr[sm], Q_ptr[sm + 1]):
            nbr = tmpl_n[bm, Q_slot[k]]
            if nbr >= 0:
                p_minus[f, nbr] += np.float32(Q_w[k])
    return p_plus, p_minus


def marginal_transition_metrics(chain: Chain) -> Dict[str, np.ndarray]:
    """Compute marginal 26-neighbour transition probabilities ``(Nt, 26)``, ``codi`` ``(Nt,)``,
    ``coasy`` ``(Nt,)``, ``p_absorb`` ``(Nt,)``, and ``dead_mean`` ``(Nt,)`` directly from ``chain``.
    """
    Nt = int(chain.t_lin.size)
    p_nbr, p_abs, p_dead = _marginal_stencil_numba(
        chain.Q_ptr,
        chain.Q_slot,
        np.asarray(chain.Q_w, dtype=np.float64),
        chain.R_ptr,
        np.asarray(chain.R_ind, dtype=np.int32),
        np.asarray(chain.R_val, dtype=np.float64),
        np.asarray(chain.t_lin, dtype=np.int64),
        np.asarray(chain.a_lin, dtype=np.int64),
        np.asarray(chain.noff, dtype=np.int64),
        np.asarray(chain.dead, dtype=np.float64),
        chain.tmpl_n,
        int(chain.nb),
        Nt,
    )
    # Normalize rows that have positive continuation mass for compositional metrics (CoDI & CoAsy)
    row_sum = p_nbr.sum(axis=1, keepdims=True)
    p_norm = np.divide(p_nbr, row_sum, out=np.full_like(p_nbr, 1.0 / 26.0), where=row_sum > 1e-12)
    eps = 1e-7
    p_safe = np.maximum(p_norm, eps)
    log_p = np.log(p_safe)
    clr = log_p - log_p.mean(axis=1, keepdims=True)
    codi = np.sqrt(np.sum(clr * clr, axis=1)).astype(np.float32)
    codi[row_sum[:, 0] <= 1e-12] = 0.0

    # Antipodal asymmetry across the 13 opposite neighbour pairs: OFFSETS[25 - n] == -OFFSETS[n]
    p_rev = p_safe[:, ::-1]
    diff_anti = np.log(p_safe[:, :13]) - np.log(p_rev[:, :13])
    coasy = np.sqrt(np.mean(diff_anti * diff_anti, axis=1)).astype(np.float32)
    coasy[row_sum[:, 0] <= 1e-12] = 0.0

    return {
        "p_neighbor": p_nbr,
        "p_absorb": p_abs,
        "dead_mean": p_dead,
        "codi": codi,
        "coasy": coasy,
    }


def _voxel_to_world_unit(vectors: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Rotate unit vectors from voxel axes into scanner RAS+ world axes."""
    if len(vectors) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    R = np.asarray(affine, dtype=np.float64)[:3, :3]
    R_norm = R / np.maximum(np.linalg.norm(R, axis=0, keepdims=True), 1e-12)
    w = np.asarray(vectors, dtype=np.float64) @ R_norm.T
    nrm = np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-12)
    return (w / nrm).astype(np.float32)


def _format_odx_entry(folder: str, name: str, arr: np.ndarray) -> Tuple[str, bytes]:
    arr = np.ascontiguousarray(arr)
    if arr.ndim == 1:
        ncols = 1
    elif arr.ndim == 2:
        ncols = int(arr.shape[1])
    else:
        raise ValueError(f"ODX array {folder}/{name} must be 1D or 2D, got {arr.ndim}D")
    dt_name = _DTYPE_TO_ODX[arr.dtype]
    return f"{folder}/{name}.{ncols}.{dt_name}", arr.tobytes(order="C")


def save_transitions_odx(
    chain: Chain,
    path: Union[str, Path],
    affine: Optional[np.ndarray] = None,
    fingerprint: Optional[str] = None,
    sh_rows: Optional[np.ndarray] = None,
    lobes: Optional[Lobes] = None,
    launches: Optional[Launches] = None,
    whole_dc: Optional[np.ndarray] = None,
    surv: Optional[np.ndarray] = None,
    mittens_maps: Optional[Dict[str, np.ndarray]] = None,
    quantize_weights: bool = True,
) -> Path:
    """Persist ``chain`` (including PTT / 26-class stencil, fixel lobes, and per-voxel transition
    metrics) as a spec-compliant ``.odx`` archive (or ``.odxd`` directory).

    Parameters
    ----------
    chain : mpat.Chain
        Built Markov chain.
    path : str or Path
        Destination ``.odx`` archive or ``.odxd`` directory.
    affine : (4, 4) array, optional
        Voxel-to-RAS+mm affine (defaults to ``diag(zooms, 1)``).
    fingerprint : str, optional
        Input + config SHA-1 digest used by :meth:`Chain.build` cache validation.
    sh_rows : (Nt, K) float32 array, optional
        Transient WM spherical harmonic coefficients (if ``lobes`` is not supplied, lobes are
        segmented from ``sh_rows``).
    lobes : mpat.fixels.Lobes, optional
        Pre-segmented fODF lobes for the ``Nt`` transient WM voxels.
    launches : mpat.solve.Launches, optional
        Precomputed seed half-walk launches to embed for instant downstream solving.
    whole_dc : (Nt, 26) float32 array, optional
        Unconditioned whole-brain class visitation density for AFD attribution.
    surv : (Ns,) or (Nt,) float32 array, optional
        Overlength survival probabilities per state or per voxel.
    mittens_maps : dict, optional
        Additional per-voxel maps from ``mittens.TransitionModel.fit``.
    quantize_weights : bool
        Store ``Q_w`` and ``R_val`` in ``float32`` instead of ``float64`` (halves file size while
        preserving transition probabilities to ``< 1e-7``).
    """
    path = Path(path)
    shape = tuple(int(s) for s in chain.shape)
    Nt = int(chain.t_lin.size)
    nb = int(chain.nb)

    if affine is None:
        affine = getattr(chain, "affine", None)
    if affine is None:
        affine = np.eye(4, dtype=np.float64)
    affine = np.asarray(affine, dtype=np.float64)
    zooms = [float(x) for x in np.linalg.norm(affine[:3, :3], axis=0)]

    # Build full-grid uint8 mask matching chain.t_lin in C-order
    t_ijk = np.stack(np.unravel_index(chain.t_lin, chain.pshape), axis=1) - 1
    mask_3d = np.zeros(shape, dtype=np.uint8)
    mask_3d[tuple(t_ijk.T)] = 1
    mask_bytes = np.ascontiguousarray(mask_3d.ravel(order="C"), dtype="<u1").tobytes()

    if lobes is None and sh_rows is not None and getattr(chain, "geo", None) is not None:
        from .fixels import segment_lobes

        lobes = segment_lobes(sh_rows, chain.geo)

    entries: Dict[str, bytes] = {}
    entries["mask.uint8"] = mask_bytes

    # 1. Fixels (offsets, directions, dpf/*)
    if lobes is not None:
        nb_peaks = int(lobes.n_fixels)
        offsets_u32 = np.empty(Nt + 1, dtype="<u4")
        offsets_u32[:Nt] = lobes.offset.astype("<u4")
        offsets_u32[Nt] = np.uint32(nb_peaks)
        dirs_ras = _voxel_to_world_unit(lobes.direction, affine).astype("<f4")
        entries["offsets.uint32"] = offsets_u32.tobytes(order="C")
        entries["directions.3.float32"] = dirs_ras.tobytes(order="C")

        for fname, data in [
            _format_odx_entry("dpf", "amplitude", lobes.peak.astype("<f4")),
            _format_odx_entry("dpf", "afd", lobes.integral.astype("<f4")),
            _format_odx_entry("dpf", "dir_vox", lobes.direction.astype("<f4")),
        ]:
            entries[fname] = data

        if lobes._w_fix is not None:
            k_wf, b_wf = _format_odx_entry("dpf", "w_class", lobes._w_fix.astype("<f4"))
            entries[k_wf] = b_wf

        # Per-fixel directional (+/-) 26-neighbour transition weights
        if getattr(chain, "geo", None) is not None and nb_peaks > 0 and lobes._frow is not None:
            verts = chain.geo.vertices
            dots = lobes.direction @ verts.T
            d_plus = np.argmax(dots, axis=1)
            d_minus = np.argmin(dots, axis=1)
            b_plus = chain.geo.vbin[d_plus].astype(np.int64)
            b_minus = chain.geo.vbin[d_minus].astype(np.int64)
            p_plus, p_minus = _fixel_half_probs_numba(
                chain.Q_ptr,
                chain.Q_slot,
                np.asarray(chain.Q_w, dtype=np.float64),
                chain.tmpl_n,
                nb,
                lobes._frow.astype(np.int64),
                b_plus,
                b_minus,
            )
            for k_f, b_f in [
                _format_odx_entry("dpf", "ptt_plus", p_plus),
                _format_odx_entry("dpf", "ptt_minus", p_minus),
            ]:
                entries[k_f] = b_f
    else:
        nb_peaks = 0
        offsets_u32 = np.zeros(Nt + 1, dtype="<u4")
        entries["offsets.uint32"] = offsets_u32.tobytes(order="C")
        entries["directions.3.float32"] = np.zeros((0, 3), dtype="<f4").tobytes(order="C")

    # 2. Per-voxel transition metrics (dpv/*)
    m_metrics = marginal_transition_metrics(chain)
    dpv_dict: Dict[str, np.ndarray] = {
        "p_neighbor": m_metrics["p_neighbor"].astype("<f4"),
        "dead": chain.dead.reshape(Nt, nb).astype("<f4"),
        "dead_mean": m_metrics["dead_mean"].astype("<f4"),
        "p_absorb": m_metrics["p_absorb"].astype("<f4"),
        "codi": m_metrics["codi"].astype("<f4"),
        "coasy": m_metrics["coasy"].astype("<f4"),
    }
    if lobes is not None and lobes._w_unass is not None:
        dpv_dict["w_unass"] = lobes._w_unass.astype("<f4")
    if surv is not None:
        surv_arr = np.asarray(surv, dtype="<f4")
        if surv_arr.size == Nt * nb:
            dpv_dict["overlength_class"] = surv_arr.reshape(Nt, nb)
            dpv_dict["overlength"] = surv_arr.reshape(Nt, nb).max(axis=1)
        elif surv_arr.size == Nt:
            dpv_dict["overlength"] = surv_arr.ravel()
    if mittens_maps:
        for mk, mv in mittens_maps.items():
            mv_arr = np.asarray(mv, dtype="<f4")
            if mv_arr.shape[0] == Nt:
                dpv_dict[mk] = mv_arr

    for name, arr in dpv_dict.items():
        k_dpv, b_dpv = _format_odx_entry("dpv", name, arr)
        entries[k_dpv] = b_dpv

    # 3. Optional SH coefficients
    header_sh_fields = {}
    if sh_rows is not None and len(sh_rows) == Nt:
        sh_f32 = np.ascontiguousarray(sh_rows, dtype="<f4")
        ncoef = int(sh_f32.shape[1])
        sh_order_sym = int(round((-3.0 + np.sqrt(1.0 + 8.0 * ncoef)) / 2.0))
        if (sh_order_sym + 1) * (sh_order_sym + 2) // 2 == ncoef:
            sh_order = sh_order_sym
            full_basis = False
        else:
            sh_order = int(round(np.sqrt(ncoef) - 1.0))
            full_basis = True
        k_sh, b_sh = _format_odx_entry("sh", "coefficients", sh_f32)
        entries[k_sh] = b_sh
        header_sh_fields = {
            "SH_ORDER": sh_order,
            "SH_BASIS": "descoteaux07",
            "SH_FULL_BASIS": full_basis,
            "SH_LEGACY": True,
            "CANONICAL_DENSE_REPRESENTATION": "sh",
        }

    # 4. Sparse 26-class PTT / Markov stencil in groups/*
    w_dtype = "<f4" if quantize_weights else "<f8"
    group_arrays: Dict[str, np.ndarray] = {
        "mpat_Q_ptr": np.asarray(chain.Q_ptr, dtype="<i8"),
        "mpat_Q_slot": np.asarray(chain.Q_slot),
        "mpat_Q_w": np.asarray(chain.Q_w, dtype=w_dtype),
        "mpat_R_ptr": np.asarray(chain.R_ptr, dtype="<i8"),
        "mpat_R_ind": np.asarray(chain.R_ind, dtype="<i4"),
        "mpat_R_val": np.asarray(chain.R_val, dtype=w_dtype),
        "mpat_a_lin": np.asarray(chain.a_lin, dtype="<i8"),
        "mpat_a_class": np.asarray(chain.a_class, dtype="<i1"),
        "mpat_label": np.asarray(chain.label, dtype="<i1"),
        "mpat_tmpl_n": np.asarray(chain.tmpl_n, dtype="<i2"),
        "mpat_tmpl_b": np.asarray(chain.tmpl_b, dtype="<i2"),
    }
    kill = getattr(chain, "kill", None)
    if kill is not None and kill.size > 0:
        group_arrays["mpat_kill"] = np.asarray(kill, dtype="<u1")
    if whole_dc is not None:
        group_arrays["mpat_whole_dc"] = np.asarray(whole_dc, dtype="<f4")
    if launches is not None:
        group_arrays.update(
            {
                "mpat_L_seeds": np.asarray(launches.seeds, dtype="<i4"),
                "mpat_L_seed_lin": np.asarray(launches.seed_lin, dtype="<i8"),
                "mpat_L_pair_seed": np.asarray(launches.pair_seed, dtype="<i4"),
                "mpat_L_pair_w": np.asarray(launches.pair_w, dtype="<f8"),
                "mpat_L_pair_d": np.asarray(launches.pair_d, dtype="<i4"),
                "mpat_L_hseed": np.asarray(launches.hseed, dtype="<i8"),
                "mpat_L_lptr": np.asarray(launches.lptr, dtype="<i8"),
                "mpat_L_lkey": np.asarray(launches.lkey, dtype="<i4"),
                "mpat_L_lval": np.asarray(launches.lval, dtype="<f8").reshape(launches.lval.shape[0], -1),
                "mpat_L_dead": np.asarray(launches.dead, dtype="<f8"),
            }
        )

    for gname, garr in group_arrays.items():
        k_g, b_g = _format_odx_entry("groups", gname, garr)
        entries[k_g] = b_g

    header = {
        "VOXEL_TO_RASMM": affine.tolist(),
        "DIMENSIONS": list(shape),
        "NB_VOXELS": Nt,
        "NB_PEAKS": nb_peaks,
        **header_sh_fields,
        "mpat": {
            "format_version": 1,
            "fingerprint": fingerprint or "",
            "config": chain.cfg.key(),
            "zooms": zooms,
            "pshape": list(chain.pshape),
            "nb": nb,
            "n_states": int(Nt * nb),
            "n_transitions": int(chain.Q_w.size),
            "absorb_classes": list(ABSORB_CLASSES),
            "neighbor_offsets_ijk": OFFSETS.tolist(),
            "neighbor_labels": neighbor_labels(affine),
            "lobes_peak_min": float(lobes.peak_min) if lobes is not None else 0.1,
            "lobes_merge_ratio": float(lobes.merge_ratio) if lobes is not None else 0.8,
        },
    }
    entries["header.json"] = json.dumps(header, indent=2).encode("utf-8")

    path.parent.mkdir(parents=True, exist_ok=True)
    if str(path).endswith(".odxd") or path.is_dir():
        path.mkdir(parents=True, exist_ok=True)
        for rel_path, content in entries.items():
            full_p = path / rel_path
            full_p.parent.mkdir(parents=True, exist_ok=True)
            full_p.write_bytes(content)
    else:
        with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
            for rel_path, content in entries.items():
                zf.writestr(rel_path, content)

    return path


def _parse_odx_entries(path: Path) -> Tuple[dict, Dict[str, Tuple[int, np.dtype, bytes]]]:
    """Read ``header.json`` and all typed array entries from an ``.odx`` archive or ``.odxd`` dir."""
    raw_files: Dict[str, bytes] = {}
    if path.is_dir():
        for p in path.rglob("*"):
            if p.is_file():
                rel = p.relative_to(path).as_posix()
                raw_files[rel] = p.read_bytes()
    else:
        with zipfile.ZipFile(path, mode="r") as zf:
            for name in zf.namelist():
                if not name.endswith("/"):
                    raw_files[name] = zf.read(name)

    header = json.loads(raw_files["header.json"].decode("utf-8"))
    parsed: Dict[str, Tuple[int, np.dtype, bytes]] = {}
    for rel, data in raw_files.items():
        if rel == "header.json":
            continue
        fname = rel.rsplit("/", 1)[-1]
        parts = fname.split(".")
        prefix = rel.rsplit("/", 1)[0] + "/" if "/" in rel else ""
        if len(parts) == 2 and parts[1] in _ODX_TO_DTYPE:
            parsed[f"{prefix}{parts[0]}"] = (1, _ODX_TO_DTYPE[parts[1]], data)
        elif len(parts) == 3 and parts[2] in _ODX_TO_DTYPE:
            parsed[f"{prefix}{parts[0]}"] = (int(parts[1]), _ODX_TO_DTYPE[parts[2]], data)
    return header, parsed


def _decode_array(entry: Tuple[int, np.dtype, bytes], squeeze_1d: bool = True) -> np.ndarray:
    ncols, dt, raw = entry
    arr = np.frombuffer(raw, dtype=dt).copy()
    if ncols > 1 or not squeeze_1d:
        return arr.reshape(-1, ncols)
    return arr


def load_transitions_odx(
    path: Union[str, Path],
    fingerprint: Optional[str] = None,
    return_extras: bool = False,
):
    """Load an :class:`mpat.Chain` (and optionally ``affine``, ``lobes``, ``launches``, ``whole_dc``,
    ``dpv``) from an ``.odx`` archive or ``.odxd`` directory written by :func:`save_transitions_odx`.
    """
    path = Path(path)
    header, entries = _parse_odx_entries(path)
    mpat_meta = header.get("mpat") or {}
    if fingerprint is not None and mpat_meta.get("fingerprint") != fingerprint:
        raise ValueError(f"ODX fingerprint mismatch in {path}")
    if "groups/mpat_Q_ptr" not in entries:
        raise ValueError(f"{path} does not contain MPAT stencil arrays (groups/mpat_Q_ptr)")

    cfg_dict = dict(mpat_meta.get("config") or {})
    valid_cfg = {f.name for f in ChainConfig.__dataclass_fields__.values()}
    cfg = ChainConfig(**{k: v for k, v in cfg_dict.items() if k in valid_cfg})

    dims = tuple(int(d) for d in header["DIMENSIONS"])
    pshape = tuple(int(d) for d in mpat_meta.get("pshape", [d + 2 for d in dims]))
    affine = np.asarray(header["VOXEL_TO_RASMM"], dtype=np.float64)
    zooms = tuple(float(z) for z in mpat_meta.get("zooms", np.linalg.norm(affine[:3, :3], axis=0)))

    mask_flat = _decode_array(entries["mask"]).reshape(dims) > 0
    trans_p = np.zeros(pshape, dtype=bool)
    trans_p[1:-1, 1:-1, 1:-1] = mask_flat

    chain = Chain()
    chain.cfg = cfg
    chain.shape = dims
    chain.pshape = pshape
    chain.affine = affine
    chain.noff = (OFFSETS @ np.array([pshape[1] * pshape[2], pshape[2], 1])).astype(np.int64)
    chain.nb = int(mpat_meta.get("nb", 26))
    chain.t_lin = np.flatnonzero(trans_p.ravel()).astype(np.int64)
    chain.tindex = np.full(int(np.prod(pshape)), -1, dtype=np.int64)
    chain.tindex[chain.t_lin] = np.arange(chain.t_lin.size)

    chain.a_lin = _decode_array(entries["groups/mpat_a_lin"]).astype(np.int64)
    chain.a_class = _decode_array(entries["groups/mpat_a_class"]).astype(np.int64)
    chain.aindex = np.full(int(np.prod(pshape)), -1, dtype=np.int64)
    chain.aindex[chain.a_lin] = np.arange(chain.a_lin.size)

    if "groups/mpat_label" in entries:
        chain.label = _decode_array(entries["groups/mpat_label"]).astype(np.int8)
    else:
        label = np.zeros(int(np.prod(pshape)), dtype=np.int8)
        label[chain.a_lin] = np.where(chain.a_class == CSF, K.BLOCKED, K.ABSORBING)
        label[chain.t_lin] = K.TRANSIENT
        chain.label = label

    chain.Q_ptr = _decode_array(entries["groups/mpat_Q_ptr"]).astype(np.int64)
    chain.Q_slot = _decode_array(entries["groups/mpat_Q_slot"])
    chain.Q_w = _decode_array(entries["groups/mpat_Q_w"]).astype(np.float64)
    chain.R_ptr = _decode_array(entries["groups/mpat_R_ptr"]).astype(np.int64)
    chain.R_ind = _decode_array(entries["groups/mpat_R_ind"]).astype(np.int32)
    chain.R_val = _decode_array(entries["groups/mpat_R_val"]).astype(np.float64)
    chain.tmpl_n = _decode_array(entries["groups/mpat_tmpl_n"], squeeze_1d=False).astype(np.int16)
    chain.tmpl_b = _decode_array(entries["groups/mpat_tmpl_b"], squeeze_1d=False).astype(np.int16)
    chain.dead = _decode_array(entries["dpv/dead"], squeeze_1d=False).astype(np.float64).ravel()

    if "groups/mpat_kill" in entries:
        chain.kill = _decode_array(entries["groups/mpat_kill"]).astype(bool)
    else:
        chain.kill = None

    chain.geo = build_geometry(zooms, cfg.angle, cfg.step, curv_sigma=cfg.curv_sigma)
    chain._finalize()

    if not return_extras:
        return chain

    extras: Dict[str, object] = {
        "affine": affine,
        "zooms": zooms,
        "header": header,
    }
    if "sh/coefficients" in entries:
        extras["sh_rows"] = _decode_array(entries["sh/coefficients"], squeeze_1d=False).astype(np.float32)

    if int(header.get("NB_PEAKS", 0)) > 0 and "offsets" in entries:
        offsets_u32 = _decode_array(entries["offsets"]).astype(np.int64)
        count = np.diff(offsets_u32)
        offset = offsets_u32[:-1]
        if "dpf/dir_vox" in entries:
            dir_vox = _decode_array(entries["dpf/dir_vox"], squeeze_1d=False).astype(np.float64)
        else:
            dirs_ras = _decode_array(entries["directions"], squeeze_1d=False).astype(np.float64)
            dir_vox = _voxel_to_world_unit(dirs_ras, np.linalg.inv(affine)).astype(np.float64)
        peak = _decode_array(entries["dpf/amplitude"]).astype(np.float64) if "dpf/amplitude" in entries else np.ones(len(dir_vox))
        integral = _decode_array(entries["dpf/afd"]).astype(np.float64) if "dpf/afd" in entries else peak.copy()
        w_fix = _decode_array(entries["dpf/w_class"], squeeze_1d=False).astype(np.float64) if "dpf/w_class" in entries else None
        w_unass = _decode_array(entries["dpv/w_unass"], squeeze_1d=False).astype(np.float64) if "dpv/w_unass" in entries else None
        frow = np.repeat(np.arange(len(count), dtype=np.int64), count)
        extras["lobes"] = Lobes(
            count=count,
            offset=offset,
            direction=dir_vox,
            peak=peak,
            integral=integral,
            peak_min=float(mpat_meta.get("lobes_peak_min", 0.1)),
            merge_ratio=float(mpat_meta.get("lobes_merge_ratio", 0.8)),
            _frow=frow,
            _w_fix=w_fix,
            _w_unass=w_unass,
        )

    if "groups/mpat_L_seeds" in entries:
        lval_2d = _decode_array(entries["groups/mpat_L_lval"], squeeze_1d=False).astype(np.float64)
        extras["launches"] = Launches(
            seeds=_decode_array(entries["groups/mpat_L_seeds"], squeeze_1d=False).astype(np.int64),
            seed_lin=_decode_array(entries["groups/mpat_L_seed_lin"]).astype(np.int64),
            pair_seed=_decode_array(entries["groups/mpat_L_pair_seed"]).astype(np.int64),
            pair_w=_decode_array(entries["groups/mpat_L_pair_w"]).astype(np.float64),
            pair_d=_decode_array(entries["groups/mpat_L_pair_d"], squeeze_1d=False).astype(np.int64),
            hseed=_decode_array(entries["groups/mpat_L_hseed"]).astype(np.int64),
            lptr=_decode_array(entries["groups/mpat_L_lptr"]).astype(np.int64),
            lkey=_decode_array(entries["groups/mpat_L_lkey"]).astype(np.int32),
            lval=lval_2d,
            dead=_decode_array(entries["groups/mpat_L_dead"]).astype(np.float64),
        )

    if "groups/mpat_whole_dc" in entries:
        extras["whole_dc"] = _decode_array(entries["groups/mpat_whole_dc"], squeeze_1d=False).astype(np.float32)

    dpv_out = {}
    for k, entry in entries.items():
        if k.startswith("dpv/"):
            dpv_out[k[4:]] = _decode_array(entry, squeeze_1d=True)
    extras["dpv"] = dpv_out
    return chain, extras


def save_transitions_nifti(
    chain: Chain,
    prefix: Union[str, Path],
    affine: Optional[np.ndarray] = None,
) -> Dict[str, str]:
    """Write classic 4-D ``(X, Y, Z, 26)`` voxel-to-neighbour transition probabilities
    (``<prefix>_desc-transitions_dwimap.nii.gz`` + ``.json``) and 3-D ``codi`` and ``coasy``
    scalar NIfTI maps.
    """
    prefix_str = str(prefix)
    if affine is None:
        affine = getattr(chain, "affine", np.eye(4))
    affine = np.asarray(affine, dtype=np.float64)
    shape = tuple(int(s) for s in chain.shape)
    t_ijk = np.stack(np.unravel_index(chain.t_lin, chain.pshape), axis=1) - 1
    idx = tuple(t_ijk.T)

    m = marginal_transition_metrics(chain)
    vol_4d = np.zeros(shape + (26,), dtype=np.float32)
    vol_4d[idx] = m["p_neighbor"]

    trans_nii = f"{prefix_str}_desc-transitions_dwimap.nii.gz"
    trans_json = f"{prefix_str}_desc-transitions_dwimap.json"
    codi_nii = f"{prefix_str}_desc-codi_map.nii.gz"
    coasy_nii = f"{prefix_str}_desc-coasy_map.nii.gz"

    Path(trans_nii).parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(vol_4d, affine), trans_nii)
    nib.save(nib.Nifti1Image(chain.to_volume(m["codi"]), affine), codi_nii)
    nib.save(nib.Nifti1Image(chain.to_volume(m["coasy"]), affine), coasy_nii)

    sidecar = {
        "Description": "Voxel-to-26-neighbour Markov transition probabilities",
        "config": chain.cfg.key(),
        "neighbor_offsets_ijk": OFFSETS.tolist(),
        "neighbor_labels": neighbor_labels(affine),
        "n_transient_voxels": int(chain.t_lin.size),
        "n_states": int(chain.t_lin.size * chain.nb),
        "n_transitions": int(chain.Q_w.size),
    }
    with open(trans_json, "w") as fh:
        json.dump(sidecar, fh, indent=2)

    return {
        "transitions": trans_nii,
        "sidecar": trans_json,
        "codi": codi_nii,
        "coasy": coasy_nii,
    }


def load_transitions_nifti(
    nifti_path_or_prefix: Union[str, Path],
    absorb_class: Optional[np.ndarray] = None,
) -> Chain:
    """Build a 26-class :class:`mpat.Chain` directly from a 4-D ``(X, Y, Z, 26)`` transition
    probability NIfTI (such as ``*_desc-transitions_dwimap.nii.gz`` or MITTENS ``*_double.nii.gz``).

    Each step from voxel ``v`` into neighbour ``n`` arrives in direction class ``n`` (``nb = 26``),
    and outgoing hops from class ``b`` are gated by the turning cone (or unconstrained when
    ``angle >= 180``).
    """
    from scipy import ndimage

    p = Path(nifti_path_or_prefix)
    meta = {}
    if not p.exists():
        for suffix in ("_desc-transitions_dwimap.nii.gz", "_double.nii.gz", ".nii.gz"):
            cand = Path(f"{nifti_path_or_prefix}{suffix}")
            if cand.exists():
                p = cand
                break
    for js_cand in (
        Path(str(p).replace(".nii.gz", ".json").replace(".nii", ".json")),
        Path(f"{nifti_path_or_prefix}_transitions.json"),
    ):
        if js_cand.exists():
            with open(js_cand) as fh:
                meta = json.load(fh)
            break

    img = nib.load(p)
    vol_4d = np.asarray(img.dataobj, dtype=np.float64)
    if vol_4d.ndim != 4 or vol_4d.shape[3] != 26:
        raise ValueError(f"Expected 4-D (X, Y, Z, 26) transition NIfTI, got shape {vol_4d.shape}")

    shape = tuple(int(d) for d in vol_4d.shape[:3])
    pshape = tuple(s + 2 for s in shape)
    affine = np.asarray(img.affine, dtype=np.float64)
    zooms = tuple(float(z) for z in np.linalg.norm(affine[:3, :3], axis=0))

    transient = np.any(vol_4d > 0, axis=-1)
    if absorb_class is None:
        dil = ndimage.binary_dilation(transient, np.ones((3, 3, 3), dtype=bool))
        absorb_class = np.where(dil & ~transient, 1, 0).astype(np.int8)
    else:
        absorb_class = np.asarray(absorb_class, dtype=np.int8)

    cfg_dict = dict(meta.get("config") or {})
    valid_cfg = {f.name for f in ChainConfig.__dataclass_fields__.values()}
    cfg = ChainConfig(**{k: v for k, v in cfg_dict.items() if k in valid_cfg})

    chain = Chain()
    chain.cfg = cfg
    chain.shape = shape
    chain.pshape = pshape
    chain.affine = affine
    chain.noff = (OFFSETS @ np.array([pshape[1] * pshape[2], pshape[2], 1])).astype(np.int64)
    chain.nb = 26

    trans_p = np.zeros(pshape, dtype=bool)
    trans_p[1:-1, 1:-1, 1:-1] = transient
    cls_p = np.zeros(pshape, dtype=np.int8)
    cls_p[1:-1, 1:-1, 1:-1] = absorb_class
    label = np.zeros(pshape, dtype=np.int8)
    label[cls_p > 0] = K.ABSORBING
    label[cls_p == CSF] = K.BLOCKED
    label[trans_p] = K.TRANSIENT
    chain.label = label.ravel()
    chain.t_lin = np.flatnonzero(trans_p.ravel()).astype(np.int64)
    chain.tindex = np.full(label.size, -1, dtype=np.int64)
    chain.tindex[chain.t_lin] = np.arange(chain.t_lin.size)
    a_mask = ndimage.binary_dilation(trans_p, np.ones((3, 3, 3), dtype=bool)) & ~trans_p
    chain.a_lin = np.flatnonzero(a_mask.ravel()).astype(np.int64)
    chain.aindex = np.full(label.size, -1, dtype=np.int64)
    chain.aindex[chain.a_lin] = np.arange(chain.a_lin.size)
    chain.a_class = cls_p.ravel()[chain.a_lin].astype(np.int64)
    chain.geo = build_geometry(zooms, cfg.angle, cfg.step, curv_sigma=cfg.curv_sigma)

    # Each outgoing hop n in 0..25 lands in arrival class n: template has 26 slots per class
    Nt = int(chain.t_lin.size)
    nb = 26
    chain.tmpl_n = np.broadcast_to(np.arange(26, dtype=np.int16)[None, :], (26, 26)).copy()
    chain.tmpl_b = np.broadcast_to(np.arange(26, dtype=np.int16)[None, :], (26, 26)).copy()

    t_ijk = np.stack(np.unravel_index(chain.t_lin, pshape), axis=1) - 1
    p_rows = np.clip(vol_4d[tuple(t_ijk.T)], 0.0, None)

    # Precompute turning cone compatibility between incoming class b and outgoing step n
    unit_off = OFFSETS * np.asarray(zooms, dtype=np.float64)
    unit_off /= np.linalg.norm(unit_off, axis=1, keepdims=True)
    cos_max = float(np.cos(np.deg2rad(max(cfg.angle, 45.0))))
    cone_ok = (unit_off @ unit_off.T) >= cos_max

    Ns = Nt * nb
    Q_ptr = np.zeros(Ns + 1, dtype=np.int64)
    R_ptr = np.zeros(Ns + 1, dtype=np.int64)
    q_slot_l, q_w_l = [], []
    r_ind_l, r_val_l = [], []
    dead = np.zeros(Ns, dtype=np.float64)

    for v_idx in range(Nt):
        v_lin = chain.t_lin[v_idx]
        p_v = p_rows[v_idx]
        for b in range(nb):
            s = v_idx * nb + b
            w_b = np.where(cone_ok[b], p_v, 0.0)
            if w_b.sum() <= 0.0:
                w_b = p_v.copy()
            # Zero out blocked (CSF) neighbours and renormalize
            for n in range(26):
                if chain.label[v_lin + chain.noff[n]] == K.BLOCKED:
                    w_b[n] = 0.0
            tot = float(w_b.sum())
            if tot <= 0.0:
                dead[s] = 1.0
                Q_ptr[s + 1] = len(q_w_l)
                R_ptr[s + 1] = len(r_val_l)
                continue
            scale = min(1.0 / tot, 1.0) if tot > 1.0 else 1.0 / tot
            w_b = w_b * scale
            dead[s] = max(0.0, 1.0 - float(w_b.sum()))
            r_map: Dict[int, float] = {}
            for n in range(26):
                w = float(w_b[n])
                if w <= 0.0:
                    continue
                nb_lin = v_lin + chain.noff[n]
                lb = chain.label[nb_lin]
                if lb == K.TRANSIENT:
                    q_slot_l.append(n)
                    q_w_l.append(w)
                elif lb == K.ABSORBING:
                    ai = int(chain.aindex[nb_lin])
                    if ai >= 0:
                        r_map[ai] = r_map.get(ai, 0.0) + w
                else:
                    dead[s] += w
            for ai, rw in sorted(r_map.items()):
                r_ind_l.append(ai)
                r_val_l.append(rw)
            Q_ptr[s + 1] = len(q_w_l)
            R_ptr[s + 1] = len(r_val_l)

    chain.Q_ptr = Q_ptr
    chain.Q_slot = np.asarray(q_slot_l, dtype=np.uint8)
    chain.Q_w = np.asarray(q_w_l, dtype=np.float64)
    chain.R_ptr = R_ptr
    chain.R_ind = np.asarray(r_ind_l, dtype=np.int32)
    chain.R_val = np.asarray(r_val_l, dtype=np.float64)
    chain.dead = dead
    chain.kill = None
    chain._finalize()
    return chain
