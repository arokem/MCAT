(bundles)=
# Bundle definitions: include, exclude and endpoint ROIs

MPAT selects bundles with pyAFQ's ROI-based bundle definitions. This page says what each kind
of ROI means for a streamline, what MPAT adds to pyAFQ's definitions, and how to check the ROIs
a run used. How the ROIs enter the solves is in {ref}`rois`.

## What a bundle streamline is

Every seed launches one streamline per fODF peak, as two **half-walks** from the seed voxel, one
in each direction ({numref}`fig-launches`). The streamline belongs to the bundle when all of these
hold (`mpat.solve`):

- it **passes every include ROI**, in either half, in any order;
- it **touches no exclude ROI**, in either half;
- **both ends are valid**: each half ends in gray matter or at the brain edge (a fiber leaving
  the field of view), not by dying in white matter or outlasting the soft maximum length
  (`max_length`, 250 mm);
- its ends satisfy the **endpoint ROIs** (the table under {ref}`bundles-endpoints`).

MPAT computes the probability of this event for every streamline at once, with no sampling. A
bundle's density counts only the streamlines that meet it, weighted by that probability.

## Where an ROI acts

Walkers move through white matter, end in gray matter or at the brain edge, and never enter
CSF (see the {ref}`tissue table <tissue-table>`). An ROI therefore acts only on the voxels a streamline can occupy:

| ROI voxels in | include | exclude | start / end |
|---|---|---|---|
| white matter | passed when a walker enters | the streamline is rejected if a walker enters | no effect |
| gray matter or edge | passed when the streamline *ends* there | the streamline is rejected if it ends there | where the endpoint test is made |
| CSF | never passed | no effect | no effect |
| the seed voxel | counts as passed | the seed launches nothing for this bundle | no effect |

Two consequences are easy to miss:

- **An endpoint ROI must reach gray matter.** Walkers end in the first gray-matter (or edge)
  voxel they enter, so only that layer of an endpoint ROI matters. An endpoint ROI that lies
  entirely in white matter or CSF can never be met, and the bundle comes out empty. This is
  why endpoint ROIs are grown before use (below).
- **An include ROI over gray matter is an endpoint requirement in disguise.** A walker cannot
  pass through gray matter, so it can only "pass" such an ROI by ending in it. If that is what
  you mean, use an endpoint ROI instead: it is grown to the gray-matter boundary, and the
  requirement is explicit.

## Exclude ROIs

A streamline that touches an exclude ROI anywhere is rejected, whichever half touches it and
however briefly. That includes ending in the ROI and being seeded in it.

- **Sealing.** Each exclude ROI is grown by its 6-neighbors before use (`mpat.afq.seal_26`).
  Walkers step to any of the 26 neighbors, so without sealing they could cross a one-voxel
  sheet diagonally. The ROI in use is therefore about one voxel thicker on each face than
  drawn.
- **Voxel-exact.** The test is made at every step of every walk. pyAFQ tests points along each
  streamline, so a short dip into an exclude ROI can pass in pyAFQ and not in MPAT. Thin
  exclusions next to a bundle cut more in MPAT. The rules below exist for this.
- **The other hemisphere.** A bundle with `cross_midline=False` also excludes the other
  hemisphere, past `midline_tol` mm from the midline (default 0). The midline is world
  $x = 0$ in the fODF image's coordinates, so it assumes data roughly aligned with the
  midline at $x = 0$ (as in AC–PC-aligned data). A bundle's side comes from its name ("Left
  …", "Right …"), or else from where its ROIs are. `no_midline` (`--no-midline`) turns this off
  for named bundles, and `midline_tol` (`--midline-tol`) overrides the tolerance of the
  bundles whose rules set one.
- **Cost.** Exclusions are free: they only mask states in the solves the bundle needs anyway.

## Include ROIs

A streamline passes an include ROI when either of its halves enters a voxel of it at least once
(in white matter, at its seed, or where it ends; see the table above). It must pass every
include ROI of the bundle.

- **No order.** A streamline that passes B and then A counts for a bundle defined "A, then B".
  Streamlines have no direction, so to tell a bundle's two ends apart, use start and end ROIs.
- **Sealing.** Include ROIs are sealed as exclude ROIs are, so a walker cannot step diagonally
  past a thin waypoint without passing it.
- **Cost.** "Passes A" is not a mask, so MPAT turns it into a signed sum of avoidance problems
  (inclusion–exclusion, {ref}`rois`). Each include ROI doubles the bundle's solution columns:
  $m$ include ROIs need $2^m$ avoid sets. Two waypoints, as most of pyAFQ's bundles have, cost
  4 times a bundle with none.

(bundles-endpoints)=
## Endpoint ROIs: start and end

Both ends of every bundle streamline must be valid (gray matter or the brain edge). Start and
end ROIs then say *where*:

| ROIs given | the streamline's ends must be | end kinds |
|---|---|---|
| neither | anywhere valid | 1 |
| start only, or end only | one end in the ROI, the other anywhere valid | 2 |
| start and end | one end in start and the other in end, either way round | 2 (3 if start and end overlap) |

A streamline is not oriented, so "start" and "end" are interchangeable names: the two halves
of a streamline may meet them either way round. Each end kind multiplies the bundle's solution
columns, so a bundle with two include ROIs and start and end ROIs needs $2^2 \times 2 = 8$.

- **Growth.** Endpoint ROIs are grown by pyAFQ's `dist_to_atlas` (`segmentation_params`, 4 mm
  by default), rounded down to whole voxels, as pyAFQ's own endpoint test does. This takes a
  template ROI drawn in or near gray matter to the gray-matter boundary where walkers end.
- **The subject's thalamus.** When pyAFQ's tissue comes from SynthSeg, an endpoint ROI that is
  the template's thalamus (at least half of it within `dist_to_atlas` of a thalamus, or its
  center inside one) is replaced by the subject's SynthSeg thalamus on its side. The thalamus
  is gray matter, where walkers end, so a template thalamus a few millimeters off would make
  most real endings invalid. `synthseg_thalamus=False` (`--mni-thalamus`) keeps the
  template's.

## MPAT's rules

Some bundles need more than pyAFQ's definition says, because MPAT applies ROIs exactly (see
"Voxel-exact" above) or because the ends are better taken from the subject's own segmentation.
These additions are **rules**, kept in `mpat.bundles`:

| rule | default | |
|---|---|---|
| `midline_tol` | 0 | mm past the midline a `cross_midline=False` bundle may go before the other hemisphere is excluded |
| `exclude_mni_boxes` | none | extra exclude ROIs: MNI boxes `{"x": (lo, hi), "y": ..., "z": ...}` in mm, warped with pyAFQ's mapping and sealed |
| `include_mni_boxes` | none | the same, as include ROIs |
| `start_synthseg`, `end_synthseg` | none | endpoint ROIs as SynthSeg labels of the subject (names of `AFQ.nn.synthseg.SynthSegLabels`, or values); they replace the template's start / end |
| `synthseg_tol_mm` | `dist_to_atlas` | how far SynthSeg endpoint ROIs are grown, mm (0: the labels as they are) |
| `thalamus` | `"synthseg"` | `"template"`: keep a template thalamus endpoint ROI (see above) |

A rule comes from, in order of precedence:

1. the bundle entry's own `"mpat"` sub-dictionary;
2. `mpat.bundles.MPAT_RULES`, by bundle name, which applies to any bundle dictionary that
   uses pyAFQ's names. It holds a 10 mm `midline_tol` for the corticospinal tracts, whose
   brainstem course runs next to the midline, and an exclusion of the fornix and septum for
   the anterior thalamic radiations (`--no-fornix-exclude` drops it);
3. the defaults above.

SynthSeg labels beyond pyAFQ's own segmentation (the cortical parcels, the hypothalamus) make
MPAT run SynthSeg's extra models once per subject; the result is kept next to the outputs and
reused. Newer pyAFQ rejects keys it does not know, so `mpat.bundles.pyafq_bd(bd)` gives the
dictionary without the `"mpat"` entries, for pyAFQ's own recognition.

**MPAT's default bundles** (`mpat.bundles.mpat_bd()`) are pyAFQ's defaults, the SLF and the
callosal bundles, plus two
bundles MPAT defines with SynthSeg ends: the hippocampal cingulum (one end in the
hippocampus) and the fornix (hippocampus to hypothalamus, through the fornix's body).
`--synthseg-cst` replaces pyAFQ's corticospinal tracts with ones defined by SynthSeg ends
only: the precentral, paracentral and postcentral cortex to the brainstem. The command line
uses `mpat_bd()` unless you pass your own `bundle_info`. From Python, `mpat.run` uses the
`ParticipantAFQ`'s bundle dictionary, pyAFQ's defaults unless you set it. For MPAT's, pass
`bundle_info=mpat_bd()` (`from mpat.bundles import mpat_bd`).

## What does not carry over from pyAFQ

MPAT computes an expectation over all streamlines, so pyAFQ's criteria that test individual
streamlines have no counterpart and are ignored: curvature, length (beyond MPAT's own soft
`max_length`), cleaning, and overlap with other bundles. pyAFQ's probability maps
(`prob_map`) and RecoBundles definitions are not used either.

```{warning}
A bundle with none of include, exclude, start and end ROIs (for example one defined only by
a `prob_map`) has no requirement but valid ends (and its hemisphere, with
`cross_midline=False`). It comes out as every streamline in the brain, or in that hemisphere.
```

## A bundle of your own

ROIs are images, as in any pyAFQ bundle dictionary. With `"space": "subject"`, they are on
the subject's grid, which MPAT resamples to the fODF's with nearest-neighbor interpolation;
otherwise they are in the template's space and warped with pyAFQ's mapping.

```python
import nibabel as nib
import AFQ.api.bundle_dict as abd
from AFQ.api.participant import ParticipantAFQ
import mpat
from mpat.bundles import mpat_bd

bd = abd.BundleDict({
    "Left My Tract": {
        "include": [nib.load("waypoint_a.nii.gz"), nib.load("waypoint_b.nii.gz")],
        "exclude": [nib.load("avoid.nii.gz")],
        "start": nib.load("frontal_end.nii.gz"),
        "cross_midline": False,
        "space": "subject",
        # MPAT only: the other end in the subject's supramarginal gyrus
        "mpat": {"end_synthseg": ["CTX_LH_SUPRAMARGINAL"]},
    },
})

afq = ParticipantAFQ(dwi, bval, bvec, t1, output_dir, bundle_info=bd)
mpat.run(afq, n_proc=8, max_mem="16GB", bundles=["Left My Tract"])
```

This bundle needs $2^2 \times 2 = 8$ solution columns: two include ROIs, and start and end.
To add it to MPAT's default bundles instead, pass `bundle_info=mpat_bd() + bd`.

## Checking the ROIs

Every run writes the ROIs each bundle used, on the fODF's grid, after warping, sealing and
growth ({doc}`outputs`):

- `..._desc-start_mask` and `_desc-end_mask`: the endpoint ROIs;
- `..._desc-include_mask`: the include ROIs, labeled 1, 2, … in the order of the definition;
- `..._desc-exclude_mask`: the exclude ROIs, without the other hemisphere.

`mpat.run(..., viz_rois=True)` (`--viz-rois`) shows the include (green) and exclude (red) ROIs
on the viewer pages. `mpat.afq.write_rois(afq)` (`--rois-only`) writes the masks, and a viewer
page of them (`<sub>_desc-mpat_rois.html`), for a finished run without solving anything. Compare `_desc-mpat_dseg` (the tissue labels) with
the endpoint masks: an endpoint ROI that does not reach gray matter (label 2) or the edge
(label 4) cannot be met.
