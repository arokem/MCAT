"""3-D branching arbors, deterministic reactive-flux flow lines (.trx), and iso-surfaces (.surf.gii).

Why 2-D Maximum Intensity Projections (MIPs) of ``voxel_afd_t`` reveal sub-bundles and offshoots
that a single 1-D centerline (``bundle_core``) misses:
1. A white-matter pathway (such as ``ARC_L``, ``CST_L``, or ``CGC_R``) has a collimated deep trunk
   that bifurcates into multiple cortical gyral patches and secondary laminar offshoots.
2. Unlike noisy Monte Carlo streamlines, MPAT's 26-neighbour reactive edge flux ``flux26(u, n)``
   defines a smooth, deterministic, bundle-conditioned 3-D vector field:
       J_b(u) = sum_{n=0}^{25} F_b(u, n) * (o_n * zooms / ||o_n * zooms||)
   whose integral curves (RK4 streamlines) follow the exact trunk, bifurcations, and offshoots
   with zero sampling jitter.

This module extracts three 3-D geometric representations for interactive visualization in TRXViz:
- ``extract_branch_arbor``: Multi-branch generalization of ``bundle_core`` that connects every
  cortical endpoint patch and lateral cross-section density sub-peak smoothly into the main trunk.
- ``extract_flux_streamlines``: Deterministic RK4 flow lines of ``J_b(x)`` stratified across the
  bundle's trunk, primary branches, and peripheral offshoots, carrying per-vertex (DPV) and
  per-streamline (DPS) scalars and TRX groups.
- ``write_trx`` & ``write_isosurface_gifti``: Direct exporters to ``.trx`` and ``.surf.gii``.
"""
from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import csgraph

from .core import OFFS, _anchor, _graph, _resample, bundle_core, split_ends
from .fields import endpoint_patches
from .geometry import OFFSETS


@dataclass
class BranchCurve:
    """A single 3-D branch curve in the bundle's branching arbor."""
    branch_id: int
    name: str
    kind: str                  # "trunk", "start_branch", "end_branch", "lateral_offshoot"
    xyz: np.ndarray            # (N, 3) RAS+ world coordinates (mm)
    afd: np.ndarray            # (N,) sampled AFD density along the branch
    q_harmonic: np.ndarray     # (N,) harmonic coordinate in [0, 1]
    coherence: np.ndarray      # (N,) directional flux coherence in [0, 1]
    k_eff: np.ndarray          # (N,) effective branching degree in [1, 26]
    share: float               # relative weight / endpoint share of this branch
    length_mm: float


@dataclass
class BranchArbor:
    """Complete 3-D branching tree (trunk + cortical fan branches + lateral offshoots)."""
    bundle_name: str
    trunk: BranchCurve
    branches: List[BranchCurve] = field(default_factory=list)


@dataclass
class FluxTractogram:
    """Deterministic 3-D reactive-flux flow lines with DPV, DPS, and TRX groups."""
    streamlines: List[np.ndarray]               # list of (N_i, 3) float32 RAS+ world coords
    dpv: Dict[str, List[np.ndarray]]            # per-vertex scalar arrays matching streamlines
    dps: Dict[str, np.ndarray]                  # per-streamline scalar arrays (N_streamlines,)
    groups: Dict[str, np.ndarray]               # group_name -> uint32 streamline indices
    affine: np.ndarray                          # (4, 4) voxel-to-RASMM affine
    dimensions: Tuple[int, int, int]            # (nx, ny, nz)


def _sample_trilinear(vol: np.ndarray, ijk: np.ndarray) -> np.ndarray:
    """Sample a 3-D volume at fractional voxel coordinates ``ijk`` ((N, 3))."""
    return ndi.map_coordinates(np.asarray(vol, dtype=np.float64), ijk.T, order=1, mode="nearest")


def extract_branch_arbor(
    afd_vol: np.ndarray,
    endpoints_vol: np.ndarray,
    affine: np.ndarray,
    q_vol: Optional[np.ndarray] = None,
    coherence_vol: Optional[np.ndarray] = None,
    keff_vol: Optional[np.ndarray] = None,
    start_mask: Optional[np.ndarray] = None,
    end_mask: Optional[np.ndarray] = None,
    bundle_name: str = "bundle",
    max_end_branches: int = 6,
    max_lateral_offshoots: int = 8,
    min_branch_sep_mm: float = 7.0,
    n_trunk_nodes: int = 100,
    n_branch_nodes: int = 60,
    smooth_vox: float = 1.3,
) -> BranchArbor:
    """Extract the 3-D multi-branch ridge arbor of a bundle.

    1. Solves the 2-pass cross-section ridge core (``bundle_core``) to get the primary trunk
       and the cross-section normalized ridge graph ``G2`` where following any local ridge
       incurred near-zero cost per mm.
    2. Segments the start and end endpoint distributions into distinct cortical patches
       (``endpoint_patches`` + local maxima separated by >= ``min_branch_sep_mm``) and traces
       each cortical patch along ``G2`` into the main trunk.
    3. Finds secondary 3-D local maxima of the AFD field that lie at least ``min_branch_sep_mm``
       away from the existing trunk/branches (capturing laminar offshoots and sub-fascicles
       visible in the MIP) and traces their ridge paths back into the trunk.
    """
    d = np.clip(np.nan_to_num(np.asarray(afd_vol, float)), 0.0, None)
    ep = np.clip(np.nan_to_num(np.asarray(endpoints_vol, float)), 0.0, None)
    if d.max() <= 0:
        raise ValueError("Empty AFD/density volume")

    dn = d / d.max()
    vox_mm = float(np.abs(np.linalg.det(affine[:3, :3])) ** (1.0 / 3.0))
    inv_aff = np.linalg.inv(affine)

    # 1. Main trunk via bundle_core
    core = bundle_core(
        d, affine, ep, start=start_mask, end=end_mask, n_nodes=n_trunk_nodes, smooth=smooth_vox
    )

    S_roi, E_roi, _ = split_ends(ep, affine, start_mask, end_mask)
    keep = dn > 1e-5
    lab, _ = ndi.label(keep, np.ones((3, 3, 3)))
    core_lab = lab[np.unravel_index(np.argmax(dn), dn.shape)]
    keep = lab == core_lab
    bod = keep & (dn >= 5e-3)

    s_vox = _anchor(ep, S_roi, bod, "centroid")
    e_vox = _anchor(ep, E_roi, bod, "centroid")

    cost1 = np.where(keep, -np.log(np.where(keep, np.maximum(dn, 1e-12), 1.0)), 0.0)
    G1, idx, lin = _graph(cost1, keep, vox_mm)
    src = np.array([lin[tuple(s_vox)]])
    dst = np.array([lin[tuple(e_vox)]])

    # Pass 1 path & cross-section ridge normalization
    dist_s, pred_s, _ = csgraph.dijkstra(G1, indices=src, min_only=True, return_predecessors=True)
    t_end = dst[0]
    p0 = [t_end]
    while pred_s[p0[-1]] >= 0:
        p0.append(pred_s[p0[-1]])
    p0 = np.array(p0[::-1])

    dist_cs, _, owner = csgraph.dijkstra(G1, indices=p0, min_only=True, return_predecessors=True)
    dv = d[tuple(idx.T)]
    ok = np.isfinite(dist_cs)
    mx = np.zeros(len(idx))
    np.maximum.at(mx, owner[ok], dv[ok])
    rel = np.zeros(len(idx))
    rel[ok] = dv[ok] / np.maximum(mx[owner[ok]], 1e-300)

    # Slight penalty off the trunk so secondary branches merge smoothly into the trunk ridge
    cost2 = np.zeros(d.shape)
    cost2[tuple(idx.T)] = np.where(ok, -np.log(np.maximum(rel, 1e-6)), 50.0)
    G2, idx2, lin2 = _graph(cost2, keep, vox_mm)

    # Reconstruct trunk voxel indices in G2
    dist_trunk, pred_trunk, _ = csgraph.dijkstra(
        G2, indices=np.array([lin2[tuple(s_vox)]]), min_only=True, return_predecessors=True
    )
    p_trunk = [lin2[tuple(e_vox)]]
    while pred_trunk[p_trunk[-1]] >= 0:
        p_trunk.append(pred_trunk[p_trunk[-1]])
    p_trunk = np.array(p_trunk[::-1])

    # Multi-source Dijkstra from the entire trunk on G2: any branch target voxel can trace its
    # minimum-ridge-cost path straight into the nearest point on the trunk!
    dist_to_trunk, pred_to_trunk, trunk_owner = csgraph.dijkstra(
        G2, indices=p_trunk, min_only=True, return_predecessors=True
    )

    def _make_branch_curve(xyz_pts: np.ndarray, bid: int, name: str, kind: str, share: float) -> BranchCurve:
        ijk_pts = nib.affines.apply_affine(inv_aff, xyz_pts)
        afd_s = _sample_trilinear(d, ijk_pts)
        q_s = (
            _sample_trilinear(q_vol, ijk_pts)
            if q_vol is not None
            else np.linspace(0.0, 1.0, len(xyz_pts))
        )
        coh_s = (
            _sample_trilinear(coherence_vol, ijk_pts)
            if coherence_vol is not None
            else np.full(len(xyz_pts), 0.7)
        )
        keff_s = (
            _sample_trilinear(keff_vol, ijk_pts)
            if keff_vol is not None
            else np.full(len(xyz_pts), 6.0)
        )
        # Orient curve from smaller q_harmonic to larger q_harmonic
        if q_s[0] > q_s[-1]:
            xyz_pts = xyz_pts[::-1].copy()
            afd_s = afd_s[::-1].copy()
            q_s = q_s[::-1].copy()
            coh_s = coh_s[::-1].copy()
            keff_s = keff_s[::-1].copy()
        seg_len = float(np.linalg.norm(np.diff(xyz_pts, axis=0), axis=1).sum())
        return BranchCurve(
            branch_id=bid,
            name=name,
            kind=kind,
            xyz=xyz_pts,
            afd=afd_s,
            q_harmonic=q_s,
            coherence=coh_s,
            k_eff=keff_s,
            share=float(share),
            length_mm=seg_len,
        )

    trunk_curve = _make_branch_curve(core.xyz, 0, f"{bundle_name}_Trunk", "trunk", 1.0)
    branches: List[BranchCurve] = [trunk_curve]
    covered_xyz = [core.xyz]

    def _is_well_separated(cand_xyz: np.ndarray, min_dist: float) -> bool:
        all_cov = np.vstack(covered_xyz)
        dists = np.linalg.norm(all_cov - cand_xyz[None, :], axis=1)
        return float(dists.min()) >= min_dist

    def _trace_voxel_to_trunk(target_ijk: np.ndarray, n_pts: int) -> Optional[np.ndarray]:
        u = int(lin2[tuple(target_ijk)])
        if u < 0 or not np.isfinite(dist_to_trunk[u]):
            return None
        path_nodes = [u]
        while pred_to_trunk[path_nodes[-1]] >= 0:
            path_nodes.append(int(pred_to_trunk[path_nodes[-1]]))
        # Also include 4 nodes along the trunk from the junction for a seamless visual Y-junction
        junc = path_nodes[-1]
        pos_in_trunk = np.flatnonzero(p_trunk == junc)
        if len(pos_in_trunk) > 0:
            ti = int(pos_in_trunk[0])
            center_ti = len(p_trunk) // 2
            step_dir = 1 if ti < center_ti else -1
            for k in range(1, 6):
                tk = ti + k * step_dir
                if 0 <= tk < len(p_trunk):
                    path_nodes.append(int(p_trunk[tk]))
        if len(path_nodes) < 4:
            return None
        vox_coords = idx2[np.array(path_nodes)].astype(float)
        xyz_raw = nib.affines.apply_affine(affine, vox_coords)
        if smooth_vox and len(xyz_raw) > 3:
            xyz_raw = ndi.gaussian_filter1d(xyz_raw, smooth_vox, axis=0, mode="nearest")
        return _resample(xyz_raw, n_pts)

    # 2. Cortical fan branches at Start and End ROIs
    cand_vox = np.argwhere(bod)
    bid_counter = 1
    for side_mask, side_kind, side_prefix in (
        (S_roi, "start_branch", "StartFan"),
        (E_roi, "end_branch", "EndFan"),
    ):
        ep_side = np.where(side_mask, ep, 0.0)
        if ep_side.sum() <= 0:
            continue
        labels, shares = endpoint_patches(ep_side, sigma_vox=1.8, min_share=0.03, prominence=0.15)
        # Also add local maxima of smoothed endpoint density so multi-gyral fans are captured
        blur_ep = ndi.gaussian_filter(ep_side, 1.5)
        local_max = (blur_ep == ndi.maximum_filter(blur_ep, size=5)) & (blur_ep > 0.05 * blur_ep.max())
        peak_ijks = np.argwhere(local_max)
        peak_vals = blur_ep[tuple(peak_ijks.T)] if len(peak_ijks) else np.array([])
        order = np.argsort(-peak_vals)

        anchors_ijk = []
        anchor_shares = []
        for p_idx in range(1, len(shares) + 1):
            patch_mask = labels == p_idx
            a_ijk = _anchor(ep_side, patch_mask, bod, "mode")
            anchors_ijk.append(a_ijk)
            anchor_shares.append(shares[p_idx - 1])
        for oi in order:
            p_ijk = peak_ijks[oi]
            nearest_bod = cand_vox[np.argmin(((cand_vox - p_ijk) ** 2).sum(1))]
            anchors_ijk.append(nearest_bod)
            anchor_shares.append(float(peak_vals[oi] / max(blur_ep.sum(), 1e-12)))

        added_side = 0
        for a_ijk, sh in zip(anchors_ijk, anchor_shares):
            if added_side >= max_end_branches:
                break
            a_xyz = nib.affines.apply_affine(affine, a_ijk.astype(float))
            if not _is_well_separated(a_xyz, min_branch_sep_mm):
                continue
            curve_xyz = _trace_voxel_to_trunk(a_ijk, n_branch_nodes)
            if curve_xyz is None:
                continue
            b_curve = _make_branch_curve(
                curve_xyz, bid_counter, f"{bundle_name}_{side_prefix}_{added_side + 1}", side_kind, sh
            )
            if b_curve.length_mm >= min_branch_sep_mm:
                branches.append(b_curve)
                covered_xyz.append(curve_xyz)
                bid_counter += 1
                added_side += 1

    # 3. Secondary lateral offshoots (3-D ridges of AFD density away from covered branches)
    blur_d = ndi.gaussian_filter(dn, 1.2)
    loc_peaks = (
        (blur_d == ndi.maximum_filter(blur_d, size=5))
        & keep
        & (blur_d >= 0.025)
    )
    pk_ijks = np.argwhere(loc_peaks)
    if len(pk_ijks) > 0:
        pk_scores = blur_d[tuple(pk_ijks.T)]
        for oi in np.argsort(-pk_scores):
            if len([b for b in branches if b.kind == "lateral_offshoot"]) >= max_lateral_offshoots:
                break
            p_ijk = pk_ijks[oi]
            p_xyz = nib.affines.apply_affine(affine, p_ijk.astype(float))
            if not _is_well_separated(p_xyz, min_branch_sep_mm * 1.15):
                continue
            curve_xyz = _trace_voxel_to_trunk(p_ijk, n_branch_nodes)
            if curve_xyz is None:
                continue
            off_idx = len([b for b in branches if b.kind == "lateral_offshoot"]) + 1
            b_curve = _make_branch_curve(
                curve_xyz,
                bid_counter,
                f"{bundle_name}_Offshoot_{off_idx}",
                "lateral_offshoot",
                float(pk_scores[oi]),
            )
            if b_curve.length_mm >= min_branch_sep_mm:
                branches.append(b_curve)
                covered_xyz.append(curve_xyz)
                bid_counter += 1

    return BranchArbor(bundle_name=bundle_name, trunk=trunk_curve, branches=branches)


def extract_flux_streamlines(
    chain: object,
    flux26: np.ndarray,
    afd_vol: np.ndarray,
    affine: np.ndarray,
    q_vol: Optional[np.ndarray] = None,
    coherence_vol: Optional[np.ndarray] = None,
    keff_vol: Optional[np.ndarray] = None,
    sift2_afd_vol: Optional[np.ndarray] = None,
    arbor: Optional[BranchArbor] = None,
    n_streamlines: int = 800,
    step_mm: float = 0.85,
    max_steps: int = 260,
    min_length_mm: float = 18.0,
    afd_mass_fraction: float = 0.96,
) -> FluxTractogram:
    """Integrate deterministic 3-D flow lines through the bundle's reactive flux field J_b(x).

    Every streamline is a smooth RK4 integral curve of the exact Markov reactive flux vector
    field (with zero Monte Carlo sampling noise), stratified across the dense trunk, primary
    cortical branches, and fine peripheral offshoots.
    """
    shape = chain.shape
    zooms = np.sqrt((affine[:3, :3] ** 2).sum(0))
    inv_aff = np.linalg.inv(affine)

    # 1. Build 3-D vector field V_ijk (in voxel-index velocity units) from 26-neighbour flux26
    f_pos = np.maximum(np.asarray(flux26, dtype=np.float64), 0.0)
    # Net flux vector in voxel-offset coordinates
    unit_offs_vox = OFFSETS / np.linalg.norm(OFFSETS * zooms[None, :], axis=1, keepdims=True)
    vec_t = f_pos @ unit_offs_vox  # (Nt, 3)

    # Also add harmonic gradient fallback where net vector cancels or near absorbing boundaries
    if q_vol is not None:
        gq = np.stack(np.gradient(np.asarray(q_vol, dtype=np.float64)), axis=-1)
        # Extract transient voxels using chain.to_volume inverse (padded transient mask)
        pad_t = np.zeros(chain.pshape, dtype=bool)
        pad_t.ravel()[chain.t_lin] = True
        t_mask = pad_t[1:-1, 1:-1, 1:-1]
        gq_t = gq[t_mask]
        v_norm = np.linalg.norm(vec_t, axis=1, keepdims=True)
        g_norm = np.linalg.norm(gq_t, axis=1, keepdims=True)
        use_g = (v_norm[:, 0] <= 1e-12) & (g_norm[:, 0] > 1e-12)
        vec_t[use_g] = gq_t[use_g]

    vec_vol = np.zeros(shape + (3,), dtype=np.float64)
    for d_ax in range(3):
        vol_d = chain.to_volume(vec_t[:, d_ax], "transient").astype(np.float64)
        # Slight Gaussian smoothing (0.55 vox) gives C^1 sub-voxel streamlines while preserving branches
        vec_vol[..., d_ax] = ndi.gaussian_filter(vol_d, 0.55)

    v_mag_vol = np.linalg.norm(vec_vol, axis=-1)
    unit_vec_vol = np.divide(
        vec_vol, v_mag_vol[..., None], out=np.zeros_like(vec_vol), where=v_mag_vol[..., None] > 1e-12
    )

    afd_clean = np.clip(np.nan_to_num(np.asarray(afd_vol, dtype=np.float64)), 0.0, None)
    peak_afd = float(afd_clean.max())
    if peak_afd <= 0:
        raise ValueError("Empty AFD volume in extract_flux_streamlines")

    # Determine support threshold enclosing afd_mass_fraction (e.g. 96% of total AFD mass)
    nz_vals = np.sort(afd_clean[afd_clean > 0])[::-1]
    csum = np.cumsum(nz_vals)
    cutoff_idx = min(int(np.searchsorted(csum, afd_mass_fraction * csum[-1])), len(nz_vals) - 1)
    afd_floor = max(float(nz_vals[cutoff_idx]) * 0.35, peak_afd * 1e-3)
    support_mask = (afd_clean >= afd_floor) & (v_mag_vol > 0)

    # 2. Select stratified deterministic seed points (ridge peaks + offshoots + weighted grid)
    cand_ijk = np.argwhere(support_mask)
    cand_afd = afd_clean[tuple(cand_ijk.T)]
    # Weight by afd^0.55 so peripheral offshoots and branches are well-represented alongside the trunk
    weights = cand_afd ** 0.55
    order = np.argsort(-weights)

    # Stratified sub-voxel jittered seeds with minimum spacing so lines fill the 3-D volume cleanly
    seeds_ijk: List[np.ndarray] = []
    if arbor is not None:
        for b in arbor.branches:
            b_ijk = nib.affines.apply_affine(inv_aff, b.xyz)
            for pt in b_ijk[::3]:
                seeds_ijk.append(pt)

    # Add deterministic Halton-like offsets around top voxels across the support
    offsets_sub = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.32, -0.25, 0.18],
            [-0.28, 0.30, -0.22],
            [0.20, 0.28, -0.30],
            [-0.25, -0.22, 0.28],
        ]
    )
    stride = max(1, len(order) // max(n_streamlines // 2, 1))
    for sub in offsets_sub:
        for idx_o in order[::stride]:
            if len(seeds_ijk) >= n_streamlines * 2:
                break
            seeds_ijk.append(cand_ijk[idx_o].astype(np.float64) + sub)
        if len(seeds_ijk) >= n_streamlines * 2:
            break

    step_vox = float(step_mm / np.mean(zooms))

    def _vel_at(pt_ijk: np.ndarray) -> np.ndarray:
        if np.any(pt_ijk < 1) or np.any(pt_ijk >= np.array(shape) - 2):
            return np.zeros(3)
        ix, iy, iz = int(round(pt_ijk[0])), int(round(pt_ijk[1])), int(round(pt_ijk[2]))
        if not support_mask[ix, iy, iz]:
            return np.zeros(3)
        # Fast trilinear lookup on unit_vec_vol
        x0, y0, z0 = int(np.floor(pt_ijk[0])), int(np.floor(pt_ijk[1])), int(np.floor(pt_ijk[2]))
        xd, yd, zd = pt_ijk[0] - x0, pt_ijk[1] - y0, pt_ijk[2] - z0
        c00 = unit_vec_vol[x0, y0, z0] * (1 - xd) + unit_vec_vol[x0 + 1, y0, z0] * xd
        c01 = unit_vec_vol[x0, y0, z0 + 1] * (1 - xd) + unit_vec_vol[x0 + 1, y0, z0 + 1] * xd
        c10 = unit_vec_vol[x0, y0 + 1, z0] * (1 - xd) + unit_vec_vol[x0 + 1, y0 + 1, z0] * xd
        c11 = unit_vec_vol[x0, y0 + 1, z0 + 1] * (1 - xd) + unit_vec_vol[x0 + 1, y0 + 1, z0 + 1] * xd
        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd
        v = c0 * (1 - zd) + c1 * zd
        nrm = float(np.linalg.norm(v))
        return v / nrm if nrm > 1e-6 else np.zeros(3)

    def _integrate_half(start_ijk: np.ndarray, sign: float) -> List[np.ndarray]:
        pts = [start_ijk.copy()]
        cur = start_ijk.copy()
        prev_dir = None
        for _ in range(max_steps):
            k1 = sign * _vel_at(cur)
            if np.linalg.norm(k1) < 1e-6:
                break
            if prev_dir is not None and float(np.dot(k1, prev_dir)) < 0.35:
                break
            k2 = sign * _vel_at(cur + 0.5 * step_vox * k1)
            if np.linalg.norm(k2) < 1e-6:
                break
            k3 = sign * _vel_at(cur + 0.5 * step_vox * k2)
            k4 = sign * _vel_at(cur + step_vox * k3)
            step_vec = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            nrm = float(np.linalg.norm(step_vec))
            if nrm < 1e-6:
                break
            step_dir = step_vec / nrm
            nxt = cur + step_vox * step_dir
            pts.append(nxt)
            prev_dir = step_dir
            cur = nxt
        return pts

    raw_streamlines_xyz: List[np.ndarray] = []
    is_arbor_flag: List[int] = []
    branch_id_list: List[int] = []

    # First include the exact BranchArbor ridge curves as hero streamlines
    if arbor is not None:
        for b in arbor.branches:
            raw_streamlines_xyz.append(b.xyz.astype(np.float32))
            is_arbor_flag.append(1)
            branch_id_list.append(int(b.branch_id))

    # Integrate deterministic RK4 flux streamlines
    for s_ijk in seeds_ijk:
        if len(raw_streamlines_xyz) >= n_streamlines:
            break
        bwd = _integrate_half(s_ijk, -1.0)[::-1]
        fwd = _integrate_half(s_ijk, +1.0)[1:]
        full_ijk = np.array(bwd + fwd, dtype=np.float64)
        if len(full_ijk) < 6:
            continue
        xyz = nib.affines.apply_affine(affine, full_ijk)
        seg_len = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
        if seg_len < min_length_mm:
            continue
        # Assign to nearest branch in arbor if available
        b_id = 0
        if arbor is not None and len(arbor.branches) > 1:
            mid_pt = xyz[len(xyz) // 2]
            end_pt1, end_pt2 = xyz[0], xyz[-1]
            best_d = 1e30
            for b in arbor.branches:
                d_min = min(
                    float(np.linalg.norm(b.xyz - mid_pt[None, :], axis=1).min()),
                    float(np.linalg.norm(b.xyz - end_pt1[None, :], axis=1).min()),
                    float(np.linalg.norm(b.xyz - end_pt2[None, :], axis=1).min()),
                )
                if d_min < best_d:
                    best_d = d_min
                    b_id = b.branch_id
        raw_streamlines_xyz.append(xyz.astype(np.float32))
        is_arbor_flag.append(0)
        branch_id_list.append(int(b_id))

    # 3. Compute DPV and DPS scalars and TRX groups
    dpv_afd: List[np.ndarray] = []
    dpv_sift2: List[np.ndarray] = []
    dpv_q: List[np.ndarray] = []
    dpv_coh: List[np.ndarray] = []
    dpv_keff: List[np.ndarray] = []
    dpv_flux: List[np.ndarray] = []

    dps_mean_afd = np.zeros(len(raw_streamlines_xyz), dtype=np.float32)
    dps_peak_afd = np.zeros(len(raw_streamlines_xyz), dtype=np.float32)
    dps_length = np.zeros(len(raw_streamlines_xyz), dtype=np.float32)
    dps_branch_id = np.array(branch_id_list, dtype=np.float32)
    dps_is_arbor = np.array(is_arbor_flag, dtype=np.float32)

    sift2_src = sift2_afd_vol if sift2_afd_vol is not None else afd_clean

    for i, xyz in enumerate(raw_streamlines_xyz):
        ijk = nib.affines.apply_affine(inv_aff, xyz.astype(np.float64))
        v_afd = _sample_trilinear(afd_clean, ijk).astype(np.float32)
        v_sift2 = _sample_trilinear(sift2_src, ijk).astype(np.float32)
        v_q = (
            _sample_trilinear(q_vol, ijk).astype(np.float32)
            if q_vol is not None
            else np.linspace(0.0, 1.0, len(xyz), dtype=np.float32)
        )
        v_coh = (
            _sample_trilinear(coherence_vol, ijk).astype(np.float32)
            if coherence_vol is not None
            else np.full(len(xyz), 0.7, dtype=np.float32)
        )
        v_keff = (
            _sample_trilinear(keff_vol, ijk).astype(np.float32)
            if keff_vol is not None
            else np.full(len(xyz), 6.0, dtype=np.float32)
        )
        v_flx = _sample_trilinear(v_mag_vol, ijk).astype(np.float32)

        dpv_afd.append(v_afd)
        dpv_sift2.append(v_sift2)
        dpv_q.append(v_q)
        dpv_coh.append(v_coh)
        dpv_keff.append(v_keff)
        dpv_flux.append(v_flx)

        dps_mean_afd[i] = float(np.mean(v_afd))
        dps_peak_afd[i] = float(np.max(v_afd))
        dps_length[i] = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())

    # Organize into TRX groups so user can toggle trunk, branches, and offshoot tiers in TRXViz
    q65 = float(np.quantile(dps_mean_afd, 0.65)) if len(dps_mean_afd) else 0.0
    q25 = float(np.quantile(dps_mean_afd, 0.25)) if len(dps_mean_afd) else 0.0

    groups: Dict[str, np.ndarray] = {
        "00_Branch_Arbor_Ridges": np.flatnonzero(dps_is_arbor > 0.5).astype(np.uint32),
        "01_Dense_Trunk_Core": np.flatnonzero((dps_is_arbor == 0) & (dps_mean_afd >= q65)).astype(np.uint32),
        "02_Primary_Branches": np.flatnonzero(
            (dps_is_arbor == 0) & (dps_mean_afd >= q25) & (dps_mean_afd < q65)
        ).astype(np.uint32),
        "03_Fine_Offshoots": np.flatnonzero((dps_is_arbor == 0) & (dps_mean_afd < q25)).astype(np.uint32),
    }
    if arbor is not None:
        for b in arbor.branches:
            members = np.flatnonzero(dps_branch_id == b.branch_id).astype(np.uint32)
            if len(members) > 0:
                groups[f"Branch_{b.branch_id:02d}_{b.name}"] = members

    return FluxTractogram(
        streamlines=raw_streamlines_xyz,
        dpv={
            "afd": dpv_afd,
            "sift2_afd": dpv_sift2,
            "q_harmonic": dpv_q,
            "coherence": dpv_coh,
            "k_eff": dpv_keff,
            "flux_magnitude": dpv_flux,
        },
        dps={
            "mean_afd": dps_mean_afd,
            "peak_afd": dps_peak_afd,
            "length_mm": dps_length,
            "branch_id": dps_branch_id,
            "is_arbor_ridge": dps_is_arbor,
        },
        groups=groups,
        affine=np.asarray(affine, dtype=np.float64),
        dimensions=(int(shape[0]), int(shape[1]), int(shape[2])),
    )


def write_trx(tractogram: FluxTractogram, out_path: str) -> str:
    """Write a ``FluxTractogram`` to a standard ``.trx`` archive compatible with ``trx-rs`` / TRXViz."""
    n_streamlines = len(tractogram.streamlines)
    lengths = np.array([len(s) for s in tractogram.streamlines], dtype=np.uint64)
    offsets = np.zeros(n_streamlines + 1, dtype=np.uint64)
    offsets[1:] = np.cumsum(lengths)
    n_vertices = int(offsets[-1])

    positions = (
        np.vstack(tractogram.streamlines).astype("<f4")
        if n_vertices > 0
        else np.zeros((0, 3), dtype="<f4")
    )

    header = {
        "DIMENSIONS": list(tractogram.dimensions),
        "VOXEL_TO_RASMM": tractogram.affine.tolist(),
        "NB_VERTICES": n_vertices,
        "NB_STREAMLINES": n_streamlines,
    }

    with zipfile.ZipFile(out_path, mode="w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("header.json", json.dumps(header, indent=2))
        zf.writestr("positions.3.float32", positions.tobytes(order="C"))
        zf.writestr("offsets.uint64", offsets.astype("<u8").tobytes(order="C"))

        for k, arr_list in tractogram.dpv.items():
            flat = (
                np.concatenate(arr_list).astype("<f4")
                if n_vertices > 0
                else np.zeros(0, dtype="<f4")
            )
            zf.writestr(f"dpv/{k}.float32", flat.tobytes(order="C"))

        for k, arr in tractogram.dps.items():
            zf.writestr(f"dps/{k}.float32", np.asarray(arr, dtype="<f4").tobytes(order="C"))

        for g_name, g_idx in tractogram.groups.items():
            safe_name = g_name.replace("/", "_").replace(" ", "_")
            zf.writestr(f"groups/{safe_name}.uint32", np.asarray(g_idx, dtype="<u4").tobytes(order="C"))

    return out_path


def write_arbor(arbor: BranchArbor, prefix: str, ref, extra: Optional[dict] = None) -> Tuple[str, str, str]:
    """Write ``<prefix>_desc-arbor_path.tsv``, ``<prefix>_desc-arbor.trx``, and
    ``<prefix>_desc-arbor.json`` for a :class:`BranchArbor`.
    """
    tsv = f"{prefix}_desc-arbor_path.tsv"
    with open(tsv, "w") as fh:
        fh.write("branch_id\tbranch_name\tkind\tnode\tx_mm\ty_mm\tz_mm\tafd\tq_harmonic\tcoherence\tk_eff\n")
        for b in arbor.branches:
            for i in range(len(b.xyz)):
                p = b.xyz[i]
                fh.write(
                    f"{b.branch_id}\t{b.name}\t{b.kind}\t{i}\t"
                    f"{p[0]:.3f}\t{p[1]:.3f}\t{p[2]:.3f}\t"
                    f"{b.afd[i]:.6g}\t{b.q_harmonic[i]:.6g}\t{b.coherence[i]:.6g}\t{b.k_eff[i]:.6g}\n"
                )

    streamlines = [b.xyz.astype(np.float32) for b in arbor.branches]
    dpv = {
        "afd": [b.afd.astype(np.float32) for b in arbor.branches],
        "q_harmonic": [b.q_harmonic.astype(np.float32) for b in arbor.branches],
        "coherence": [b.coherence.astype(np.float32) for b in arbor.branches],
        "k_eff": [b.k_eff.astype(np.float32) for b in arbor.branches],
    }
    dps = {
        "branch_id": np.array([b.branch_id for b in arbor.branches], dtype=np.float32),
        "share": np.array([b.share for b in arbor.branches], dtype=np.float32),
        "length_mm": np.array([b.length_mm for b in arbor.branches], dtype=np.float32),
    }
    groups = {
        f"Branch_{b.branch_id:02d}_{b.name}": np.array([idx], dtype=np.uint32)
        for idx, b in enumerate(arbor.branches)
    }
    tractogram = FluxTractogram(
        streamlines=streamlines,
        dpv=dpv,
        dps=dps,
        groups=groups,
        affine=np.asarray(ref.affine, dtype=np.float64),
        dimensions=tuple(int(d) for d in ref.shape[:3]),
    )
    trx = f"{prefix}_desc-arbor.trx"
    write_trx(tractogram, trx)

    js = f"{prefix}_desc-arbor.json"
    summary = {
        "method": "3-D multi-branch cross-section ridge arbor (mpat.branches)",
        "bundle": arbor.bundle_name,
        "n_branches": len(arbor.branches),
        "trunk_length_mm": round(float(arbor.trunk.length_mm), 1),
        "total_arbor_length_mm": round(float(sum(b.length_mm for b in arbor.branches)), 1),
        "branches": [
            {
                "branch_id": int(b.branch_id),
                "name": b.name,
                "kind": b.kind,
                "n_nodes": int(len(b.xyz)),
                "length_mm": round(float(b.length_mm), 1),
                "share": round(float(b.share), 4),
            }
            for b in arbor.branches
        ],
        **(extra or {}),
    }
    with open(js, "w") as fh:
        json.dump(summary, fh, indent=1)
    return tsv, trx, js


def write_arbors(folder: str, max_end_branches: int = 5, max_lateral_offshoots: int = 6) -> None:
    """Compute and save the 3-D multi-branch arbor of every bundle under ``folder``."""
    import logging
    from .core import bundle_files

    log = logging.getLogger("mpat")
    for name, f in bundle_files(folder).items():
        if f["endpoints"] is None:
            continue
        dens_path = f.get("afd") or f["density"]
        img = nib.load(dens_path)
        load_mask = lambda k: (np.asarray(nib.load(f[k]).dataobj) > 0) if f.get(k) else None  # noqa: E731
        load_vol = lambda k: np.asarray(nib.load(f[k]).dataobj, dtype=np.float32) if f.get(k) else None  # noqa: E731
        try:
            arbor = extract_branch_arbor(
                np.asarray(img.dataobj, dtype=np.float32),
                np.asarray(nib.load(f["endpoints"]).dataobj, dtype=np.float32),
                img.affine,
                q_vol=load_vol("qharmonic"),
                coherence_vol=load_vol("coherence"),
                keff_vol=load_vol("keff"),
                start_mask=load_mask("start"),
                end_mask=load_mask("end"),
                bundle_name=name,
                max_end_branches=max_end_branches,
                max_lateral_offshoots=max_lateral_offshoots,
            )
        except ValueError as err:
            log.warning("%s: no arbor (%s)", name, err)
            continue
        write_arbor(arbor, f["prefix"], img)
        log.info(
            "arbor %-24s %d branches, trunk %4.0f mm, total %4.0f mm",
            name,
            len(arbor.branches),
            arbor.trunk.length_mm,
            sum(b.length_mm for b in arbor.branches),
        )


def _taubin_smooth_mesh(
    verts: np.ndarray, faces: np.ndarray, n_iters: int = 10, lam: float = 0.5, mu: float = -0.53
) -> np.ndarray:
    """Shrink-free Taubin Laplacian mesh smoothing for organic 3-D branch iso-surfaces."""
    v = verts.copy().astype(np.float64)
    n_v = len(v)
    if n_v == 0 or len(faces) == 0:
        return v.astype(np.float32)

    edges_a = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    edges_b = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    src = np.concatenate([edges_a, edges_b])
    dst = np.concatenate([edges_b, edges_a])
    deg = np.bincount(src, minlength=n_v).astype(np.float64)
    deg_safe = np.maximum(deg, 1.0)[:, None]

    for _ in range(n_iters):
        for step_w in (lam, mu):
            nb_sum = np.zeros_like(v)
            np.add.at(nb_sum, src, v[dst])
            lap = (nb_sum / deg_safe) - v
            v = v + step_w * lap
    return v.astype(np.float32)


def write_isosurface_gifti(
    afd_vol: np.ndarray,
    affine: np.ndarray,
    out_surf_path: str,
    mass_fraction: float = 0.85,
    scalar_vol: Optional[np.ndarray] = None,
    out_func_path: Optional[str] = None,
    smooth_sigma: float = 0.65,
    taubin_iters: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract a 3-D iso-surface shell enclosing ``mass_fraction`` of the bundle's AFD mass
    and save it as a GIFTI ``.surf.gii`` mesh (with optional ``.func.gii`` vertex scalars)
    for interactive 3-D rendering in TRXViz.
    """
    from skimage.measure import marching_cubes

    vol = np.clip(np.nan_to_num(np.asarray(afd_vol, dtype=np.float64)), 0.0, None)
    if smooth_sigma > 0:
        vol = ndi.gaussian_filter(vol, smooth_sigma)
    nz = np.sort(vol[vol > 0])[::-1]
    if len(nz) == 0:
        raise ValueError("Empty volume for iso-surface")
    cs = np.cumsum(nz)
    idx = min(int(np.searchsorted(cs, mass_fraction * cs[-1])), len(nz) - 1)
    level = max(float(nz[idx]), float(vol.max()) * 1e-3)

    verts_ijk, faces, _, _ = marching_cubes(vol, level=level)
    verts_ras = nib.affines.apply_affine(affine, verts_ijk).astype(np.float32)
    verts_smooth = _taubin_smooth_mesh(verts_ras, faces, n_iters=taubin_iters)

    # Ensure outward-pointing triangle winding when det(affine) < 0
    if np.linalg.det(affine[:3, :3]) < 0:
        faces = faces[:, [0, 2, 1]]

    da_coords = nib.gifti.GiftiDataArray(
        data=verts_smooth.astype(np.float32),
        intent="NIFTI_INTENT_POINTSET",
        datatype="NIFTI_TYPE_FLOAT32",
    )
    da_faces = nib.gifti.GiftiDataArray(
        data=faces.astype(np.int32),
        intent="NIFTI_INTENT_TRIANGLE",
        datatype="NIFTI_TYPE_INT32",
    )
    gii_surf = nib.gifti.GiftiImage(darrays=[da_coords, da_faces])
    nib.save(gii_surf, out_surf_path)

    if scalar_vol is not None and out_func_path is not None:
        inv_aff = np.linalg.inv(affine)
        v_ijk = nib.affines.apply_affine(inv_aff, verts_smooth)
        vals = _sample_trilinear(scalar_vol, v_ijk).astype(np.float32)
        da_val = nib.gifti.GiftiDataArray(
            data=vals,
            intent="NIFTI_INTENT_SHAPE",
            datatype="NIFTI_TYPE_FLOAT32",
        )
        nib.save(nib.gifti.GiftiImage(darrays=[da_val]), out_func_path)

    return verts_smooth, faces.astype(np.int32)


def write_trxviz_project(
    out_project_path: str,
    bundle_nifti_path: str,
    trx_path: Optional[str] = None,
    t1w_nifti_path: Optional[str] = None,
    surface_shells: Optional[List[dict]] = None,
    bundle_surface: bool = False,
    bundle_surface_voxel_size_mm: float = 1.6,
    bundle_surface_threshold: float = 0.8,
    bundle_surface_smooth_sigma: float = 1.1,
    bundle_surface_opacity: float = 0.22,
    bundle_surface_outline_thickness: float = 2.5,
    bundle_surface_color_mode: str = "SourceColors",
    render_3d_mode: str = "Mip",
    colormap: str = "Inferno",
    vol_opacity: float = 0.82,
    window_center: float = 0.22,
    window_width: float = 0.44,
    t1w_render_3d_mode: str = "Off",
    t1w_opacity: float = 1.0,
    slice_visible: Optional[List[bool]] = None,
    slice_positions_ras: Optional[List[float]] = None,
    dpv_field: str = "q_harmonic",
    streamline_colormap: str = "Viridis",
    tube_radius_mm: float = 0.55,
    camera_target: Optional[List[float]] = None,
    azimuth_deg: float = -68.0,
    elevation_deg: float = 20.0,
    distance: float = 172.0,
) -> str:
    """Write a ``.trxviz`` workflow project file combining 3-D GPU volume rendering
    (``Mip`` or ``DirectVolume``), 3-D ``.trx`` branch arbor tubes, translucent
    cross-sectional surfaces (GIFTI iso-shells and/or ``BundleSurfaceBuild``), and
    an optional background ``T1w`` anatomical volume with slice contour lines.
    """
    if camera_target is None:
        camera_target = [-32.0, -18.0, 14.0]
    if slice_visible is None:
        slice_visible = [True, True, True] if t1w_nifti_path else [False, False, False]
    if slice_positions_ras is None:
        slice_positions_ras = [2.0, -28.0, -32.0]

    nodes = {}
    wires = []
    assets = []
    next_asset_id = 1
    next_node_id = 1

    # 1. Optional background T1w volume
    if t1w_nifti_path is not None:
        t1_asset_id = next_asset_id
        next_asset_id += 1
        assets.append({"Volume": {"id": t1_asset_id, "path": str(t1w_nifti_path)}})
        n_src = next_node_id
        n_disp = next_node_id + 1
        next_node_id += 2
        nodes[str(n_src)] = {
            "node": {
                "uuid": n_src,
                "op": {"VolumeSource": {"source_id": t1_asset_id}},
                "label": "T1w Source",
            },
            "pos": {"x": 40.0, "y": 40.0},
        }
        nodes[str(n_disp)] = {
            "node": {
                "uuid": n_disp,
                "op": {
                    "VolumeDisplay": {
                        "colormap": "Grayscale",
                        "opacity": float(t1w_opacity),
                        "window_center": 0.52,
                        "window_width": 0.92,
                        "render_3d_mode": t1w_render_3d_mode,
                    }
                },
                "label": "T1w Display",
            },
            "pos": {"x": 320.0, "y": 40.0},
        }
        wires.append({"from": {"node": n_src, "output": 0}, "to": {"node": n_disp, "input": 0}})

    # 2. Bundle 3D Volume (MIP or DirectVolume)
    b_asset_id = next_asset_id
    next_asset_id += 1
    assets.append({"Volume": {"id": b_asset_id, "path": str(bundle_nifti_path)}})
    n_bsrc = next_node_id
    n_bdisp = next_node_id + 1
    next_node_id += 2
    nodes[str(n_bsrc)] = {
        "node": {
            "uuid": n_bsrc,
            "op": {"VolumeSource": {"source_id": b_asset_id}},
            "label": "Bundle Volume Source",
        },
        "pos": {"x": 40.0, "y": 200.0},
    }
    nodes[str(n_bdisp)] = {
        "node": {
            "uuid": n_bdisp,
            "op": {
                "VolumeDisplay": {
                    "colormap": colormap,
                    "opacity": float(vol_opacity),
                    "window_center": float(window_center),
                    "window_width": float(window_width),
                    "render_3d_mode": render_3d_mode,
                }
            },
            "label": "Bundle 3D Volume Display",
        },
        "pos": {"x": 320.0, "y": 200.0},
    }
    wires.append({"from": {"node": n_bsrc, "output": 0}, "to": {"node": n_bdisp, "input": 0}})

    # 3. Optional 3-D Branch Arbor / Flux Streamlines (.trx) + optional BundleSurfaceBuild
    if trx_path is not None:
        s_asset_id = next_asset_id
        next_asset_id += 1
        assets.append({"Streamlines": {"id": s_asset_id, "path": str(trx_path), "imported": False}})
        n_ssrc = next_node_id
        n_scol = next_node_id + 1
        n_sdisp = next_node_id + 2
        next_node_id += 3
        nodes[str(n_ssrc)] = {
            "node": {
                "uuid": n_ssrc,
                "op": {"StreamlineSource": {"source_id": s_asset_id}},
                "label": "Branch Arbor Source",
            },
            "pos": {"x": 40.0, "y": 360.0},
        }
        nodes[str(n_scol)] = {
            "node": {
                "uuid": n_scol,
                "op": {"ColorByDPV": {"field": dpv_field, "colormap": streamline_colormap}},
                "label": "Color By DPV",
            },
            "pos": {"x": 320.0, "y": 360.0},
        }
        nodes[str(n_sdisp)] = {
            "node": {
                "uuid": n_sdisp,
                "op": {
                    "StreamlineDisplay": {
                        "enabled": True,
                        "render_style": "Tubes",
                        "tube_radius_mm": float(tube_radius_mm),
                        "tube_sides": 8,
                        "slab_half_width_mm": 2.5,
                        "opacity": 0.96,
                    }
                },
                "label": "Branch Arbor Tubes",
            },
            "pos": {"x": 600.0, "y": 360.0},
        }
        wires.append({"from": {"node": n_ssrc, "output": 0}, "to": {"node": n_scol, "input": 0}})
        wires.append({"from": {"node": n_scol, "output": 0}, "to": {"node": n_sdisp, "input": 0}})

        if bundle_surface:
            n_bsbuild = next_node_id
            n_bsdisp = next_node_id + 1
            next_node_id += 2
            nodes[str(n_bsbuild)] = {
                "node": {
                    "uuid": n_bsbuild,
                    "op": {
                        "BundleSurfaceBuild": {
                            "per_group": False,
                            "build_mode": "MarchingCubes",
                            "voxel_size_mm": float(bundle_surface_voxel_size_mm),
                            "threshold": float(bundle_surface_threshold),
                            "smooth_sigma": float(bundle_surface_smooth_sigma),
                            "min_component_volume_mm3": 10.0,
                            "tube_radius_mm": float(tube_radius_mm),
                            "tube_sides": 8,
                            "opacity": float(bundle_surface_opacity),
                        }
                    },
                    "label": "Branch Cross-Section Surface Build",
                },
                "pos": {"x": 600.0, "y": 500.0},
            }
            nodes[str(n_bsdisp)] = {
                "node": {
                    "uuid": n_bsdisp,
                    "op": {
                        "BundleSurfaceDisplay": {
                            "color_mode": bundle_surface_color_mode,
                            "outline_thickness": float(bundle_surface_outline_thickness),
                        }
                    },
                    "label": "Branch Cross-Section Surface Display",
                },
                "pos": {"x": 880.0, "y": 500.0},
            }
            wires.append({"from": {"node": n_scol, "output": 0}, "to": {"node": n_bsbuild, "input": 0}})
            wires.append({"from": {"node": n_bsbuild, "output": 0}, "to": {"node": n_bsdisp, "input": 0}})

    # 4. Optional GIFTI Cross-Sectional Iso-Surfaces (.surf.gii) with slice contour outlines
    if surface_shells:
        for idx_sh, sh in enumerate(surface_shells):
            surf_path = sh["path"]
            surf_label = sh.get("label", f"Cross-Section Shell {idx_sh + 1}")
            surf_color = [float(c) for c in sh.get("color", [0.15, 0.90, 1.0])]
            surf_opacity = float(sh.get("opacity", 0.20))
            outline_color = [float(c) for c in sh.get("outline_color", surf_color)]
            outline_thickness = float(sh.get("outline_thickness", 2.6))

            g_asset_id = next_asset_id
            next_asset_id += 1
            assets.append({"Surface": {"id": g_asset_id, "path": str(surf_path)}})

            n_gsrc = next_node_id
            n_gstack = next_node_id + 1
            n_gdisp = next_node_id + 2
            next_node_id += 3
            y_pos = 640.0 + idx_sh * 140.0
            nodes[str(n_gsrc)] = {
                "node": {
                    "uuid": n_gsrc,
                    "op": {"SurfaceSource": {"source_id": g_asset_id}},
                    "label": f"{surf_label} Source",
                },
                "pos": {"x": 40.0, "y": y_pos},
            }
            nodes[str(n_gstack)] = {
                "node": {
                    "uuid": n_gstack,
                    "op": {"SurfaceOverlayStack": {}},
                    "label": f"{surf_label} Overlay",
                },
                "pos": {"x": 320.0, "y": y_pos},
            }
            nodes[str(n_gdisp)] = {
                "node": {
                    "uuid": n_gdisp,
                    "op": {
                        "SurfaceDisplay": {
                            "color": surf_color,
                            "opacity": surf_opacity,
                            "outline_color": outline_color,
                            "outline_thickness": outline_thickness,
                            "show_projection_map": False,
                            "map_opacity": 1.0,
                            "map_threshold": 0.0,
                            "gloss": 0.55,
                            "projection_colormap": "Inferno",
                            "range_min": 0.0,
                            "range_max": 1.0,
                            "space": "Anatomical",
                        }
                    },
                    "label": f"{surf_label} Display",
                },
                "pos": {"x": 600.0, "y": y_pos},
            }
            wires.append({"from": {"node": n_gsrc, "output": 0}, "to": {"node": n_gstack, "input": 0}})
            wires.append({"from": {"node": n_gstack, "output": 0}, "to": {"node": n_gdisp, "input": 0}})

    project = {
        "version": 1,
        "document": {
            "next_node_uuid": next_node_id + 5,
            "graph": {"nodes": nodes, "wires": wires},
            "assets": assets,
            "camera_3d": {
                "target": [float(x) for x in camera_target],
                "azimuth_deg": float(azimuth_deg),
                "elevation_deg": float(elevation_deg),
                "distance": float(distance),
            },
            "slice_view_3d": {
                "visible": [bool(v) for v in slice_visible],
                "positions_ras": [float(p) for p in slice_positions_ras],
            },
            "slice_view_ui": {
                "mode": "Ortho",
                "single_view": "Sagittal",
                "lightbox_axis": "Axial",
                "lightbox_rows": 3,
                "lightbox_cols": 3,
                "active_axis": 2,
                "ortho_show_row": True,
                "slice_cameras": [
                    {"center": [float(camera_target[0]), float(camera_target[1])], "half_extent": 85.0, "rotation": 0.0},
                    {"center": [float(camera_target[0]), float(camera_target[2])], "half_extent": 85.0, "rotation": 0.0},
                    {"center": [float(camera_target[1]), float(camera_target[2])], "half_extent": 85.0, "rotation": 0.0},
                ],
            },
        },
    }
    with open(out_project_path, "w") as f:
        json.dump(project, f, indent=2)
    return out_project_path


def render_trxviz(
    project_path: str,
    out_png_path: str,
    width: int = 1280,
    height: int = 960,
    view: str = "3d",
    trxviz_cli_bin: Optional[str] = None,
    return_ipython_image: bool = True,
):
    """Render a ``.trxviz`` project file on the GPU via ``trxviz-cli render`` and
    optionally return an ``IPython.display.Image`` for inline display in Jupyter notebooks.
    """
    import os
    import shutil
    import subprocess
    from pathlib import Path

    # A TRXViz checkout next to this repo (../odx/TRXViz) is the last resort.
    sibling = Path(__file__).resolve().parents[3] / "odx" / "TRXViz" / "target" / "release" / "trxviz-cli"
    cli = (
        trxviz_cli_bin
        or os.environ.get("TRXVIZ_CLI")
        or shutil.which("trxviz-cli")
        or (str(sibling) if sibling.is_file() else "trxviz-cli")
    )
    cmd = [
        cli,
        "render",
        "--project",
        str(project_path),
        "--out",
        str(out_png_path),
        "--width",
        str(width),
        "--height",
        str(height),
        "--view",
        str(view),
    ]
    subprocess.run(cmd, check=True)
    if return_ipython_image:
        try:
            from IPython.display import Image as IPyImage

            return IPyImage(filename=str(out_png_path))
        except ImportError:
            pass
    return out_png_path
