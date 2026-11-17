"""Tests for ODX (.odx) and NIfTI (.nii.gz) transition probability persistence and 3-D arbors."""
from pathlib import Path

import nibabel as nib
import numpy as np

from mpat import (
    Chain,
    ChainConfig,
    extract_branch_arbor,
    prepare_launches,
    quantify_bundle,
    segment_bundles,
    segment_lobes,
    write_arbor,
)


def _sh(directions, kappa=20.0):
    from dipy.core.sphere import Sphere
    from dipy.data import get_sphere
    from dipy.reconst.shm import sh_to_sf_matrix

    sph = get_sphere(name="repulsion724")
    _, inv = sh_to_sf_matrix(Sphere(xyz=sph.vertices), sh_order_max=8, return_inv=True)
    sf = sum(np.exp(kappa * (sph.vertices @ np.asarray(d, float)) ** 2) for d in directions)
    return (sf @ inv).astype(np.float32)


def test_transitions_odx_and_nifti_roundtrip(tmp_path: Path):
    nx, ny, nz = 12, 10, 5
    zooms = (2.0, 2.0, 2.0)
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

    cfg = ChainConfig.ptt(angle=35.0, curv_sigma=18.0, probe_radius=0.5, stop_mass=1e-4)
    odx_cache = tmp_path / "chain_cache.odx"
    chain = Chain.build(sh, zooms, transient, absorb, cfg, cache=str(odx_cache))
    assert odx_cache.exists()

    # Verify cache hit from .odx
    chain_cached = Chain.build(sh, zooms, transient, absorb, cfg, cache=str(odx_cache))
    assert chain_cached.n_transitions == chain.n_transitions

    seeds = np.argwhere(transient)
    launches = prepare_launches(chain, seeds)
    t_ijk = np.stack(np.unravel_index(chain.t_lin, chain.pshape), 1) - 1
    sh_rows = sh[tuple(t_ijk.T)]
    lobes = segment_lobes(sh_rows, chain.geo)
    whole = segment_bundles(chain, [{"name": "whole"}], launches, tol=1e-6)["whole"]

    # Save full ODX with fixel lobes, launches, and whole_dc
    odx_full = tmp_path / "sub-01_desc-transitions_dwimap.odx"
    chain.save_odx(
        odx_full,
        affine=affine,
        sh_rows=sh_rows,
        lobes=lobes,
        launches=launches,
        whole_dc=whole.density_class,
        quantize_weights=True,
    )
    assert odx_full.exists()

    # Load directly from .odx without needing sh or recomputing launches/lobes
    chain_loaded, extras = Chain.load_odx(odx_full, return_extras=True)
    assert "lobes" in extras
    assert "launches" in extras
    assert "whole_dc" in extras
    assert "codi" in extras["dpv"]
    assert "coasy" in extras["dpv"]
    assert "p_neighbor" in extras["dpv"]
    assert extras["dpv"]["p_neighbor"].shape == (chain.t_lin.size, 26)

    start_mask = np.zeros((nx, ny, nz), dtype=bool)
    start_mask[1, 3:-3, 1:-1] = True
    end_mask = np.zeros((nx, ny, nz), dtype=bool)
    end_mask[-2, 3:-3, 1:-1] = True
    spec = {"name": "LeftArcuate", "start": start_mask, "end": end_mask, "include": []}

    res_orig = segment_bundles(chain, [spec], launches, tol=1e-6)["LeftArcuate"]
    res_odx = segment_bundles(chain_loaded, [spec], extras["launches"], tol=1e-6)["LeftArcuate"]
    np.testing.assert_allclose(res_odx.density_t, res_orig.density_t, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(res_odx.endpoints_a, res_orig.endpoints_a, rtol=1e-4, atol=1e-5)

    qb = quantify_bundle(
        chain_loaded,
        res_odx,
        extras["whole_dc"],
        sh_rows,
        affine,
        lobes=extras["lobes"],
        compute_boundary_conductance=False,
    )
    assert qb.tract_afd_volume_mm3 > 0.0

    # Test 3-D multi-branch arbor extraction and .trx/.json/.tsv export
    afd_vol = chain_loaded.to_volume(qb.voxel_afd_t)
    ep_vol = chain_loaded.to_volume(res_odx.endpoints_a, "absorbing")
    q_vol = chain_loaded.to_volume(qb.q_harmonic_t)
    coh_vol = chain_loaded.to_volume(qb.flux_coherence_t)
    keff_vol = chain_loaded.to_volume(qb.branching_k_eff_t)

    arbor = extract_branch_arbor(
        afd_vol,
        ep_vol,
        affine,
        q_vol=q_vol,
        coherence_vol=coh_vol,
        keff_vol=keff_vol,
        start_mask=start_mask,
        end_mask=end_mask,
        bundle_name="LeftArcuate",
    )
    ref_img = nib.Nifti1Image(afd_vol, affine)
    tsv_p, trx_p, js_p = write_arbor(arbor, str(tmp_path / "sub-01_bundle-LeftArcuate"), ref_img)
    assert Path(tsv_p).exists()
    assert Path(trx_p).exists()
    assert Path(js_p).exists()

    # Test classic 4-D (X, Y, Z, 26) NIfTI transition persistence & loading
    nii_prefix = tmp_path / "sub-01"
    nii_out = chain.save_nifti(nii_prefix, affine=affine)
    assert Path(nii_out["transitions"]).exists()
    assert Path(nii_out["sidecar"]).exists()
    assert Path(nii_out["codi"]).exists()
    assert Path(nii_out["coasy"]).exists()

    chain_from_nii = Chain.from_nifti(nii_out["transitions"], absorb_class=absorb)
    assert chain_from_nii.t_lin.size == chain.t_lin.size
    res_nii = segment_bundles(chain_from_nii, [spec], launches, tol=1e-5)["LeftArcuate"]
    assert res_nii.info["expected_bundle_streamlines"] > 0.5
