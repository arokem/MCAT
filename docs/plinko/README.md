# MPAT explained with a Plinko board

`plinko_video.py` renders a ~2-minute educational video (1280 × 720, mp4) that builds MPAT's
main algorithm (forward pass, backward pass, and their product) on a Plinko (Galton) board.

```bash
python docs/plinko/plinko_video.py              # -> docs/plinko/output/mpat_plinko.mp4  (~1 min to render)
python docs/plinko/plinko_video.py --gif        # animated GIF instead (no ffmpeg needed)
python docs/plinko/plinko_video.py --quick      # fewer balls at 12 fps: a fast draft
python docs/plinko/plinko_video.py --preview    # stills every 2 s of every scene, no video
```

It needs only numpy, matplotlib and imageio-ffmpeg, all in MPAT's `docs` extra
(`pip install -e ".[docs]"`). It takes the palette and style from
`docs/animations/phantom.py`, so it looks like the other figures in the docs. Rendered files go to
`output/`, which git ignores.

## The story

| scene | on the board | in MPAT |
|---|---|---|
| 1 · fair board | balls bounce left/right with p = ½ and pile up in a bell curve (binomial ≈ Gaussian) | Monte Carlo tracking |
| 1 · …or push the probability | the same curve with no balls, pushed down one row at a time | power iteration, $x_{t+1} = Q^\top x_t$ |
| 2 · leaning board | every peg has its own p: where a ball goes next depends only on its peg | the absorbing Markov chain on (voxel, heading) states, with transitions $Q$ from the fODF and MITTENS |
| 2 · forward pass | two streams; $S$ = P(ball passes each peg) and the flow through each gap | occupancy $S = (I - Q^\top)^{-1}\pi_0$ |
| 3 · a bundle | which balls land in the target cup? $S$ counts hits and misses alike | a bundle defined by an end ROI |
| 4 · backward pass | $\beta = p\,\beta_\text{right} + (1-p)\,\beta_\text{left}$, from the cup up | completion $\beta = (I - Q)^{-1} r$ |
| 5 · occupancy × completion | $\rho = S\cdot\beta$ keeps only the visits of balls that land in the cup | the bundle density (reactive density) |
| 6 · the check | dropped balls that hit the cup converge to $\rho$, with $1/\sqrt{N}$ error | `docs/animations/anim_mc_vs_exact.py` |
| summary | the dictionary from the board to MPAT | |

Every number on screen is computed exactly on the board (`forward`, `backward` and `edge_flow`
in the script), and the script asserts that the two passes agree: every row of $\rho$ sums to
P(cup) $= \beta$ at the top peg. The lean of the pegs is set in `leaning_board()` and the cup in
`TARGET`; change either and the whole video follows.

The analogy simplifies MPAT in a few ways, which the summary scene points out. Every ball on the
board takes exactly 12 steps, while walkers in the brain can wander for hundreds, so MPAT sums
each series until the remaining mass falls below `tol`. A real streamline also has two halves,
and include and exclude ROIs add more forward and backward columns, all through the same
products with $Q$ and $Q^\top$. See `docs/source/algorithm.md` for the full method.
