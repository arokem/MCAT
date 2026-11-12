# MPAT: Markov probabilistic absorbing tractography

MPAT computes white-matter bundles as **exact expectations over infinitely many streamlines**,
with no sampling.

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} The chain
Probabilistic tracking is treated as an absorbing Markov chain on (white-matter voxel,
direction) states. The transitions are the analytic hop geometry of
[MITTENS](https://github.com/mattcieslak/MITTENS), with the walker's incoming direction
remembered between voxels.
:::

:::{grid-item-card} Bundle selection
Include, exclude and endpoint ROIs are applied with forward and backward passes over that
chain. This is [transition path theory](https://deeptime-ml.github.io/latest/notebooks/tpt.html)'s reactive density, with inclusion–exclusion for the
include ROIs.
:::

:::{grid-item-card} pyAFQ
[pyAFQ](https://tractometry.org/pyAFQ) supplies everything else: the fODF it would track on,
the tissue PVE, the seeds (WM/GM interface), the bundle definitions, and their ROIs warped to
the subject.
:::
::::

```{animation} reactive
:plain:
:alt: A bundle emerging as the product of forward occupancy and backward completion

A bundle is the product of two exact passes over the chain: where walkers go (forward) and
whether they go on to meet the bundle's requirements (backward). See {doc}`algorithm`.
```

## Quick start

```python
from AFQ.api.participant import ParticipantAFQ      # or GroupAFQ: every participant in turn
import mpat

afq = ParticipantAFQ(dwi, bval, bvec, t1, output_dir)
mpat.run(afq, n_proc=8, max_mem="16GB")              # -> <output_dir>/mpat/
mpat.view(f"{output_dir}/mpat", t1=t1_masked, bundles=["LeftArcuate", "LeftCorticospinal"], glyphs=True)
```

or, from the command line, pyAFQ and MPAT on one participant:

```bash
mpat sub-01_dwi.nii.gz sub-01_dwi.bval sub-01_dwi.bvec sub-01_T1w.nii.gz out/ --n-proc 8 --max-mem 16GB
```

```{toctree}
:maxdepth: 2
:caption: User guide

install
usage
example_hbn
options
outputs
viewer
```

```{toctree}
:maxdepth: 2
:caption: The method

algorithm
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
cli
references
```
