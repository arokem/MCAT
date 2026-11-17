"""MPAT: Markov probabilistic absorbing tractography.

Bundle densities as exact expectations over infinitely many streamlines: the probabilistic
tracker is an absorbing Markov chain on (white-matter voxel, direction) states whose
transitions come from MITTENS's analytic path geometry, and bundle selection (include /
exclude / endpoint ROIs) is solved with forward and backward passes over that chain.
pyAFQ provides the fODF, tissue, seeds, bundle definitions and warped ROIs.

    from AFQ.api.participant import ParticipantAFQ
    import mpat
    afq = ParticipantAFQ(dwi, bval, bvec, t1, output_dir)
    mpat.run(afq, n_proc=8, max_mem="16GB")            # -> <output_dir>/mpat/
    mpat.view(f"{output_dir}/mpat", bundles=["LeftArcuate"])   # in a notebook
"""
import os

os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from .afq import run
from .branches import (
    BranchArbor,
    BranchCurve,
    FluxTractogram,
    extract_branch_arbor,
    extract_flux_streamlines,
    render_trxviz,
    write_arbor,
    write_arbors,
    write_isosurface_gifti,
    write_trx,
    write_trxviz_project,
)
from .chain import Chain, ChainConfig
from .connectome import ConnectomeResult, build_connectome
from .core import bundle_core
from .fields import BundleFields, endpoint_patches, flux_orientation, restrict_endpoints
from .fixels import DirectedFixelChain, fixel_streamlines, segment_lobes, write_fixel_directory
from .geometry import class_chord_lengths
from .quantify import BundleQuantification, afd_tract_metrics, quantify_bundle, sift2_optimize_launches
from .solve import prepare_launches, segment_bundles, solve_bundle_cascade
from .transitions import (
    load_transitions_nifti,
    load_transitions_odx,
    marginal_transition_metrics,
    save_transitions_nifti,
    save_transitions_odx,
)
from .viewer import save_html


def view(*args, **kwargs):
    """ipyniivue widget with bundle densities and cores (see mpat.viewer.view)."""
    from .viewer import view as _view

    return _view(*args, **kwargs)


__all__ = [
    "BranchArbor",
    "BranchCurve",
    "BundleFields",
    "BundleQuantification",
    "Chain",
    "ChainConfig",
    "ConnectomeResult",
    "DirectedFixelChain",
    "FluxTractogram",
    "afd_tract_metrics",
    "build_connectome",
    "bundle_core",
    "class_chord_lengths",
    "endpoint_patches",
    "extract_branch_arbor",
    "extract_flux_streamlines",
    "fixel_streamlines",
    "flux_orientation",
    "load_transitions_nifti",
    "load_transitions_odx",
    "marginal_transition_metrics",
    "prepare_launches",
    "quantify_bundle",
    "render_trxviz",
    "restrict_endpoints",
    "run",
    "save_html",
    "save_transitions_nifti",
    "save_transitions_odx",
    "segment_bundles",
    "segment_lobes",
    "sift2_optimize_launches",
    "solve_bundle_cascade",
    "view",
    "write_arbor",
    "write_arbors",
    "write_fixel_directory",
    "write_isosurface_gifti",
    "write_trx",
    "write_trxviz_project",
]
