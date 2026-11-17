"""Tests for the TRXViz <-> MPAT + MITTENS bridge (mpat.trxviz_bridge)."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from mpat.trxviz_bridge import cmd_build_transitions, cmd_solve_bundle


def _sh(directions, kappa=20.0):
    from dipy.core.sphere import Sphere
    from dipy.data import get_sphere
    from dipy.reconst.shm import sh_to_sf_matrix

    sph = get_sphere(name="repulsion724")
    _, inv = sh_to_sf_matrix(Sphere(xyz=sph.vertices), sh_order_max=8, return_inv=True)
    sf = sum(np.exp(kappa * (sph.vertices @ np.asarray(d, float)) ** 2) for d in directions)
    return (sf @ inv).astype(np.float32)


def test_trxviz_bridge_transitions_and_custom_rois(tmp_path: Path):
    nx, ny, nz = 12, 10, 5
    zooms = np.array([2.0, 2.0, 2.0], dtype=np.float64)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    sh = np.zeros((nx, ny, nz, 45), dtype=np.float32)
    transient = np.zeros((nx, ny, nz), dtype=bool)
    absorb = np.zeros((nx, ny, nz), dtype=np.int8)

    sh_x = _sh([(1.0, 0.0, 0.0)], kappa=18.0)
    transient[2:-2, 3:-3, 1:-1] = True
    absorb[1, 3:-3, 1:-1] = 1
    absorb[-2, 3:-3, 1:-1] = 1
    absorb[2:-2, 2, 1:-1] = 1
    absorb[2:-2, -3, 1:-1] = 1

    for i in range(1, nx - 1):
        for j in range(2, ny - 2):
            for k in range(1, nz - 1):
                sh[i, j, k] = sh_x

    subj_npz = tmp_path / "subj_sub01.npz"
    seeds = np.argwhere(transient)
    np.savez(
        subj_npz,
        sh=sh,
        affine=affine,
        zooms=zooms,
        transient=transient,
        absorb=absorb,
        seeds=seeds,
    )

    trans_out = tmp_path / "transitions"
    req_trans = tmp_path / "req_trans.json"
    req_trans.write_text(
        json.dumps(
            {
                "out_dir": str(trans_out),
                "working_dir": str(subj_npz),
                "preset": "ptt",
                "angle_deg": 35.0,
                "step_vox": 1.0,
                "double_odf": True,
                "stop_mass_tau": 1e-4,
                "curv_sigma_deg": 18.0,
                "probe_radius_vox": 0.0,
                "max_length_mm": 120.0,
            }
        )
    )
    cmd_build_transitions(str(req_trans))

    res_trans = json.loads((trans_out / "transitions_result.json").read_text())
    assert Path(res_trans["state_path"]).name.startswith("mpat_transitions_")
    assert len(res_trans["fp_hash"]) == 16
    expected_trans_keys = {
        "codi_double",
        "codi_single",
        "coasy",
        "p_terminate_double",
        "p_terminate_single",
        "total_amplitude",
        "kl_single",
        "chain_dead",
    }
    assert set(res_trans["volumes"].keys()) == expected_trans_keys
    for k, p in res_trans["volumes"].items():
        assert Path(p).exists(), f"Missing transition volume {k}"
        vol = np.asarray(nib.load(p).dataobj, dtype=np.float32)
        assert vol.shape == (nx, ny, nz)
    # p_terminate_double should be positive inside mask since tau > 0
    p_term = np.asarray(nib.load(res_trans["volumes"]["p_terminate_double"]).dataobj)
    assert float(p_term[transient].max()) > 0.0

    # Verify automatic hash-based cache hit on identical inputs + parameters
    trans_out_cached = tmp_path / "transitions_cached"
    req_trans_cached = tmp_path / "req_trans_cached.json"
    req_trans_cached.write_text(
        json.dumps(
            {
                "out_dir": str(trans_out_cached),
                "working_dir": str(subj_npz),
                "preset": "ptt",
                "angle_deg": 35.0,
                "step_vox": 1.0,
                "double_odf": True,
                "stop_mass_tau": 1e-4,
                "curv_sigma_deg": 18.0,
                "probe_radius_vox": 0.0,
                "max_length_mm": 120.0,
            }
        )
    )
    cmd_build_transitions(str(req_trans_cached))
    res_cached = json.loads((trans_out_cached / "transitions_result.json").read_text())
    assert res_cached["fp_hash"] == res_trans["fp_hash"]
    assert res_cached["state_path"] == res_trans["state_path"]
    assert "Cached [mpat_transitions_" in res_cached["summary"]

    # Verify direct loading when working_dir points directly to mpat_transitions_<hash>.npz
    trans_out_direct = tmp_path / "transitions_direct"
    req_trans_direct = tmp_path / "req_trans_direct.json"
    req_trans_direct.write_text(
        json.dumps(
            {
                "out_dir": str(trans_out_direct),
                "working_dir": res_trans["state_path"],
            }
        )
    )
    cmd_build_transitions(str(req_trans_direct))
    res_direct = json.loads((trans_out_direct / "transitions_result.json").read_text())
    assert res_direct["state_path"] == res_trans["state_path"]
    assert "Loaded [mpat_transitions_" in res_direct["summary"]

    # Prebuilt transitions keep their settings: asking for others is an error, not silently ignored
    req_other = tmp_path / "req_other.json"
    req_other.write_text(
        json.dumps({"out_dir": str(tmp_path / "other"), "working_dir": res_trans["state_path"],
                    "preset": "ptt", "angle_deg": 60.0})
    )
    with pytest.raises(ValueError, match="angle"):
        cmd_build_transitions(str(req_other))

    # Prepare start, end, include, avoid, and interior absorption ROI masks
    start_mask = np.zeros((nx, ny, nz), dtype=np.uint8)
    start_mask[1, 3:-3, 1:-1] = 1
    end_mask = np.zeros((nx, ny, nz), dtype=np.uint8)
    end_mask[-2, 3:-3, 1:-1] = 1
    inc_mask = np.zeros((nx, ny, nz), dtype=np.uint8)
    inc_mask[5:7, 3:-3, 1:-1] = 1

    start_p = tmp_path / "start.nii"
    end_p = tmp_path / "end.nii"
    inc_p = tmp_path / "inc.nii"
    nib.save(nib.Nifti1Image(start_mask, affine), str(start_p))
    nib.save(nib.Nifti1Image(end_mask, affine), str(end_p))
    nib.save(nib.Nifti1Image(inc_mask, affine), str(inc_p))

    # 1. Baseline solve with start, end, and include
    solve_out1 = tmp_path / "solve_base"
    req_solve1 = tmp_path / "req_solve1.json"
    req_solve1.write_text(
        json.dumps(
            {
                "state_path": res_trans["state_path"],
                "out_dir": str(solve_out1),
                "bundle_name": "PhantomTract",
                "include_mask_paths": [str(inc_p)],
                "start_mask_path": str(start_p),
                "end_mask_path": str(end_p),
                "seal_26": False,
                "midline_exclusion": "none",
                "n_streamlines": 60,
                "extract_arbor": False,
                "step_vox": 0.5,
                "min_length_mm": 4.0,
            }
        )
    )
    cmd_solve_bundle(str(req_solve1))
    res_s1 = json.loads((solve_out1 / "solve_result.json").read_text())
    assert res_s1["expected_streamlines"] > 1.0
    assert Path(res_s1["trx_path"]).exists()
    # Streamline extraction errors used to be swallowed, leaving an empty TRX
    assert not any("flux streamline extraction" in w for w in res_s1["warnings"]), res_s1["warnings"]
    assert res_s1["n_flux_streamlines"] > 0
    for k in ("afd", "density", "tdimm", "qharmonic", "coherence", "keff", "endpoints"):
        assert Path(res_s1["volumes"][k]).exists()
    mmap_file = solve_out1 / "live_density.mmap"
    assert mmap_file.exists()
    raw_hdr = mmap_file.read_bytes()[:40]
    import struct
    magic, ver, seq, phase, step, m_nx, m_ny, m_nz, resid, max_val = struct.unpack("<8I2f", raw_hdr)
    assert magic == 0x5441504D and ver == 1
    assert seq >= 2 and phase == 3
    assert (m_nx, m_ny, m_nz) == (nx, ny, nz)
    assert max_val > 0.0

    # 2. Add an Avoid ROI blocking the middle of the bundle -> expected streamlines should drop sharply
    avoid_mask = np.zeros((nx, ny, nz), dtype=np.uint8)
    avoid_mask[5:7, :, :] = 1
    avoid_p = tmp_path / "avoid.nii"
    nib.save(nib.Nifti1Image(avoid_mask, affine), str(avoid_p))

    solve_out2 = tmp_path / "solve_avoid"
    req_solve2 = tmp_path / "req_solve2.json"
    req_solve2.write_text(
        json.dumps(
            {
                "state_path": res_trans["state_path"],
                "out_dir": str(solve_out2),
                "bundle_name": "PhantomTractAvoid",
                "start_mask_path": str(start_p),
                "end_mask_path": str(end_p),
                "exclude_mask_path": str(avoid_p),
                "seal_26": True,
                "midline_exclusion": "none",
                "n_streamlines": 40,
            }
        )
    )
    cmd_solve_bundle(str(req_solve2))
    res_s2 = json.loads((solve_out2 / "solve_result.json").read_text())
    assert res_s2["expected_streamlines"] < 1e-4

    # 3. Add an arbitrary interior Absorption ROI (term_mask) at i=6..7 (inside transient WM!)
    term_mask = np.zeros((nx, ny, nz), dtype=np.uint8)
    term_mask[6:8, 3:-3, 1:-1] = 1
    term_p = tmp_path / "term.nii"
    nib.save(nib.Nifti1Image(term_mask, affine), str(term_p))

    solve_out3 = tmp_path / "solve_term"
    req_solve3 = tmp_path / "req_solve3.json"
    req_solve3.write_text(
        json.dumps(
            {
                "state_path": res_trans["state_path"],
                "out_dir": str(solve_out3),
                "bundle_name": "PhantomTractTerm",
                "start_mask_path": str(start_p),
                "term_mask_path": str(term_p),
                "seal_26": False,
                "midline_exclusion": "none",
                "n_streamlines": 40,
            }
        )
    )
    cmd_solve_bundle(str(req_solve3))
    res_s3 = json.loads((solve_out3 / "solve_result.json").read_text())
    assert res_s3["expected_streamlines"] > 1.0
    ep_vol3 = np.asarray(nib.load(res_s3["volumes"]["endpoints"]).dataobj)
    # Verify walkers absorbed inside the interior term_mask (i=6..7)
    assert float(ep_vol3[6:8, 3:-3, 1:-1].sum()) > 0.5


def test_trxviz_bridge_inputs_are_not_guessed(tmp_path: Path):
    from mpat.trxviz_bridge import _discover_subject_inputs

    nx, ny, nz = 8, 6, 4
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    sh = np.zeros((nx, ny, nz, 45), dtype=np.float32)
    sh[1:-1, 1:-1, 1:-1] = _sh([(1.0, 0.0, 0.0)])
    transient = np.zeros((nx, ny, nz), dtype=bool)
    transient[2:-2, 2:-2, 1:-1] = True
    absorb = np.where(sh.any(-1) & ~transient, 1, 0).astype(np.int8)

    # A folder's chain cache (sorted first here) is not mistaken for the staged subject
    folder = tmp_path / "subject"
    folder.mkdir()
    np.savez(folder / "sub-01_desc-chain_cache.npz", Q_ptr=np.zeros(3))
    np.savez(folder / "sub-01_staged.npz", sh=sh, affine=affine, zooms=np.array([2.0, 2.0, 2.0]),
             transient=transient, absorb=absorb)
    assert np.array_equal(_discover_subject_inputs(str(folder), None)[3], transient)

    # An fODF without tissue maps is an error, not a guess at which voxels are white matter
    bare = tmp_path / "bare"
    bare.mkdir()
    nib.save(nib.Nifti1Image(sh, affine), str(bare / "sub-01_model-csd_dwimap.nii.gz"))
    req = tmp_path / "req.json"
    req.write_text(json.dumps({"out_dir": str(tmp_path / "out"), "working_dir": str(bare)}))
    with pytest.raises(ValueError, match="No tissue maps"):
        cmd_build_transitions(str(req))
