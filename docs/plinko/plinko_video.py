"""MPAT explained with a Plinko (Galton) board: an educational video.

    python docs/plinko/plinko_video.py                  # -> docs/plinko/output/mpat_plinko.mp4
    python docs/plinko/plinko_video.py --gif            # no ffmpeg needed
    python docs/plinko/plinko_video.py --preview        # stills of every scene, no video
    python docs/plinko/plinko_video.py --quick          # fewer balls and a lower frame rate

The story, in nine scenes:

1. Title.
2. A fair board. Balls bounce left or right with probability 1/2 at every peg and pile up in a
   bell curve (binomial(12, 1/2), close to a Gaussian).
3. The same distribution with no balls: push the probability down one row at a time
   (a forward pass, x_{t+1} = Q^T x_t).
4. A leaning board. Every peg has its own probability p of sending a ball right. The next
   bounce depends only on the current peg: a Markov chain, as MPAT's tracking is.
5. Balls on the leaning board give a different (here, two-stream) distribution, and the forward
   pass reproduces it exactly: S, the probability that a ball visits each peg.
6. The bundle question: which balls end in the target cup? S counts every ball, hits and misses.
7. The backward pass: beta = P(a ball at this peg ends in the cup), filled in from the bottom up,
   beta = p beta(right) + (1 - p) beta(left), i.e. beta = (I - Q)^{-1} r.
8. Occupancy x completion: rho = S beta keeps only the visits of balls that end in the cup. This
   is MPAT's bundle density (transition path theory's reactive density).
9. A check (drop balls, keep the ones that land in the cup: their visits converge to rho),
   then the dictionary from the board to MPAT.

Everything shown is computed exactly on the board, as MPAT computes it on the brain: the board's
pegs are the chain's transient states, its bins the absorbing states, a peg's lean is its row of
Q, and the two passes are the same products with Q^T and Q that ``mpat.chain.Chain`` runs. The
board needs no MPAT install: only numpy and matplotlib (and imageio-ffmpeg for mp4), which are in
MPAT's ``docs`` extra. The colours and style are the docs' (``docs/animations/phantom.py``).
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "animations"))
import phantom as P  # noqa: E402  (the docs' shared palette, colour maps and matplotlib style)

# ------------------------------------------------------------------ the board
R = 12                      # rows of pegs; R + 1 bins
DY = 0.86                   # vertical distance between rows (1 = horizontal peg spacing)
H_BINS = 3.6                # height of the bins
Y_BIN_TOP = -R * DY + 0.3
Y_FLOOR = Y_BIN_TOP - H_BINS
DROP = np.array([0.0, 1.0])
TARGET = np.zeros(R + 1, bool)
TARGET[[9, 10]] = True      # the target cup: the bundle's end ROI

PEG_BASE = "#d9d6ce"        # a peg with nothing on it
BALL = "#3b3a37"
MISS = "#c9c6bd"
CUP = P.AQUA


def peg_xy(k, j):
    """Centre of peg j (0..k) in row k (0..R-1); row R is the bins."""
    k, j = np.broadcast_arrays(np.asarray(k, float), np.asarray(j, float))
    return np.stack([j - k / 2, -k * DY], axis=-1)


def bin_x(j):
    return np.asarray(j, float) - R / 2


PEGS = [(k, j) for k in range(R) for j in range(k + 1)]          # flat order of the pegs
PK = np.array([k for k, _ in PEGS])
PJ = np.array([j for _, j in PEGS])
PXY = peg_xy(PK, PJ)
NP = len(PEGS)


def flat(a):
    """(R, R) lower-triangular board array -> (NP,) in PEGS order."""
    return a[PK, PJ]


def fair_board():
    return np.full((R, R), 0.5)


def leaning_board():
    """p[k, j] = P(a ball at peg (k, j) bounces right).

    The top six rows push balls away from the middle (two streams); below, a gentle current
    carries balls right on the right and back towards the middle on the far left.  In MPAT the
    lean of a state comes from the fODF and the hop geometry (``mpat.kernel``).
    """
    p = np.full((R, R), np.nan)
    for k, j in PEGS:
        x = j - k / 2
        if k < 6:
            p[k, j] = 0.5 + 0.40 * np.tanh(1.3 * x)
        else:
            p[k, j] = 0.5 + 0.25 * np.tanh(x - 0.5) - 0.1 * np.exp(-(((x + 2.5) / 1.2) ** 2))
    return p


def forward(p):
    """Forward pass. S[k, j] = P(a ball visits peg (k, j)); bins[j] = P(it ends in bin j).

    Row by row this is x_{t+1} = Q^T x_t, the series MPAT sums by power iteration
    (S = sum_t (Q^T)^t pi_0).  On the board every ball takes exactly R steps.
    """
    S = np.zeros((R, R))
    S[0, 0] = 1.0
    bins = np.zeros(R + 1)
    for k in range(R):
        nxt = S[k + 1] if k < R - 1 else bins
        for j in range(k + 1):
            nxt[j] += S[k, j] * (1 - p[k, j])
            nxt[j + 1] += S[k, j] * p[k, j]
    return S, bins


def backward(p, target):
    """Backward pass. beta[k, j] = P(a ball at peg (k, j) ends in a target bin).

    beta = p beta(right child) + (1 - p) beta(left child): the products with Q, beta = sum_t Q^t r.
    """
    beta = np.zeros((R, R))
    below = target.astype(float)
    for k in range(R - 1, -1, -1):
        for j in range(k + 1):
            beta[k, j] = (1 - p[k, j]) * below[j] + p[k, j] * below[j + 1]
        below = beta[k]
    return beta


def edges():
    """Every peg's two outgoing edges: (peg index, left/right, start xy, end xy, child index)."""
    out = []
    for i, (k, j) in enumerate(PEGS):
        for side in (0, 1):
            if k < R - 1:
                end = peg_xy(k + 1, j + side)
            else:
                end = np.array([bin_x(j + side), Y_BIN_TOP])
            out.append((i, side, PXY[i], end, j + side))
    return out


EDGES = edges()
E_PEG = np.array([e[0] for e in EDGES])
E_SIDE = np.array([e[1] for e in EDGES])
E_ROW = PK[E_PEG]
E_SEGS = np.array([[e[2], e[3]] for e in EDGES])


def edge_flow(p, S, beta=None, target=None):
    """Expected number of balls crossing each edge; with beta, only those that end in the cup.

    The bundle's edge flow S(s) Q(s -> s') beta(s') is what MPAT writes as the edge flux.
    """
    out = np.empty(len(EDGES))
    for n, (i, side, _, _, child) in enumerate(EDGES):
        k, j = PEGS[i]
        q = p[k, j] if side else 1 - p[k, j]
        w = 1.0
        if beta is not None:
            w = target[child] if k == R - 1 else beta[k + 1, child]
        out[n] = S[k, j] * q * w
    return out


def simulate(p, n, rng):
    """Drop n balls: (n, R + 1) column of each ball in every row; column R is its bin."""
    cols = np.zeros((n, R + 1), int)
    for k in range(R):
        right = rng.random(n) < p[k, cols[:, k]]
        cols[:, k + 1] = cols[:, k] + right
    return cols


# ------------------------------------------------------------------ the video's stage
def to_rgba(c):
    from matplotlib.colors import to_rgba as _to_rgba

    return _to_rgba(c)


def lerp(a, b, t):
    return a + (b - a) * np.clip(t, 0, 1)


def ease(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


class Stage:
    """The figure and its persistent artists; scenes only update them."""

    def __init__(self):
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.colors import Normalize, PowerNorm
        from matplotlib.patches import Rectangle

        P.style()
        self.plt = plt
        self.fig = fig = plt.figure(figsize=(12.8, 7.2))
        ax = self.ax = fig.add_axes([0.01, 0.02, 0.56, 0.96])
        ax.set_axis_off()
        ax.set_xlim(-R / 2 - 0.9, R / 2 + 0.9)
        ax.set_ylim(Y_FLOOR - 0.35, DROP[1] + 0.6)
        ax.set_aspect("equal")

        # bins
        for j in range(R + 2):
            x = bin_x(j) - 0.5
            ax.plot([x, x], [Y_FLOOR, Y_BIN_TOP], color=P.MUTED, lw=1.2, zorder=2,
                    solid_capstyle="butt")
        ax.plot([bin_x(0) - 0.5, bin_x(R) + 0.5], [Y_FLOOR, Y_FLOOR], color=P.INK2, lw=1.6, zorder=2)
        self.bars = [ax.add_patch(Rectangle((bin_x(j) - 0.42, Y_FLOOR), 0.84, 0, lw=0, zorder=1,
                                            fc=P.MUTED)) for j in range(R + 1)]
        self.cup = ax.add_patch(Rectangle((bin_x(TARGET.nonzero()[0][0]) - 0.5, Y_FLOOR),
                                          TARGET.sum(), H_BINS, fill=False, ec=CUP, lw=3.2, zorder=6,
                                          visible=False))
        self.cup_fill = ax.add_patch(Rectangle(self.cup.get_xy(), TARGET.sum(), H_BINS, lw=0,
                                               fc=CUP, alpha=0.0, zorder=0))
        self.cup_label = P.label(ax, (bin_x(TARGET.nonzero()[0][-1]) + 0.65, Y_FLOOR + H_BINS - 0.9),
                                 "target\ncup", color=CUP, fontsize=12, fontweight="bold")
        self.cup_label.set_visible(False)
        self.beta_label = P.label(ax, (bin_x(TARGET.nonzero()[0].mean()), Y_BIN_TOP - 0.45), "β = 1",
                                  color=CUP, fontsize=13, fontweight="bold", ha="center", va="center")
        self.beta_label.set_visible(False)
        self.exact = LineCollection([], colors=P.INK, linewidths=2.4, zorder=7, capstyle="round")
        ax.add_collection(self.exact)
        (self.curve,) = ax.plot([], [], color=P.INK, lw=2.4, zorder=7)

        # flows (forward = orange, the bundle on top), lean glyphs, pegs, balls
        self.ghost = LineCollection(E_SEGS, colors=MISS, linewidths=0, zorder=3, capstyle="round")
        self.flow = LineCollection(E_SEGS, colors=P.ORANGE, linewidths=0, zorder=3.5, capstyle="round")
        ax.add_collection(self.ghost)
        ax.add_collection(self.flow)
        self.glyphs = LineCollection([], colors=P.INK2, linewidths=2.6, zorder=4, capstyle="round")
        ax.add_collection(self.glyphs)
        self.pegs = ax.scatter(PXY[:, 0], PXY[:, 1], s=170, c=[PEG_BASE], edgecolors=P.INK2,
                               linewidths=0.9, zorder=5)
        self.balls = ax.scatter([], [], s=46, c=[BALL], edgecolors="white", linewidths=0.5, zorder=8)
        ax.plot([DROP[0]], [DROP[1] + 0.15], marker="v", ms=13, color=P.INK2, zorder=5)
        self.drop_label = P.label(ax, (0.45, DROP[1] + 0.02), "drop", color=P.INK2, fontsize=11)

        # colour bar in the board's empty top-left corner
        self.cax = fig.add_axes([0.045, 0.86, 0.15, 0.022])
        self.cax.set_axis_off()
        self.cbar = self.cax.imshow(np.linspace(0, 1, 256)[None], aspect="auto", cmap=P.mass_cmap(),
                                    extent=(0, 1, 0, 1))
        self.cbar_lo = fig.text(0.045, 0.835, "0", fontsize=10, color=P.INK2, va="top")
        self.cbar_hi = fig.text(0.195, 0.835, "1", fontsize=10, color=P.INK2, va="top", ha="right")
        self.cbar_label = fig.text(0.045, 0.895, "", fontsize=11, color=P.INK2, va="bottom")
        self.show_cbar(None)

        # the narration
        X0 = 0.6
        self.step = fig.text(X0, 0.93, "", fontsize=12, color=P.MUTED, fontweight="bold", va="top")
        self.title = fig.text(X0, 0.895, "", fontsize=21, color=P.INK, va="top", fontweight="bold")
        self.body = fig.text(X0, 0.8, "", fontsize=13.5, color=P.INK2, va="top", linespacing=1.45)
        self.eq = fig.text(X0, 0.44, "", fontsize=17, color=P.INK, va="top")
        self.col_a = fig.text(X0, 0.79, "", fontsize=12.5, color=P.INK, va="top", linespacing=1.75)
        self.col_b = fig.text(X0 + 0.14, 0.79, "", fontsize=12.5, color=P.INK2, va="top",
                              linespacing=1.75)
        self.texts = [self.step, self.title, self.body, self.eq, self.col_a, self.col_b]

        # mini boards: S x beta = rho
        self.minis = []
        for n, x in enumerate((0.6, 0.735, 0.87)):
            a = fig.add_axes([x, 0.06, 0.115, 0.26])
            a.set_axis_off()
            a.set_aspect("equal")
            a.set_xlim(-R / 2 - 0.6, R / 2 + 0.6)
            a.set_ylim(-(R - 1) * DY - 0.6, 0.6)
            sc = a.scatter(PXY[:, 0], PXY[:, 1], s=13, c=[PEG_BASE], lw=0)
            t = a.set_title(["S  (forward)", "β  (backward)", "ρ = S·β"][n], fontsize=12, loc="center",
                            color=P.INK2)
            self.minis.append((a, sc, t))
        self.ops = [fig.text(0.725, 0.19, "×", fontsize=22, color=P.INK2, ha="center"),
                    fig.text(0.86, 0.19, "=", fontsize=22, color=P.INK2, ha="center")]
        self.show_minis(False)

        # Monte Carlo check: empirical vs exact rho per peg
        self.mc = fig.add_axes([0.66, 0.08, 0.24, 0.3])
        self.mc.set_xlabel("exact ρ (MPAT)", fontsize=11)
        self.mc.set_ylabel("balls that hit the cup", fontsize=11)
        self.mc.tick_params(labelsize=9)
        self.mc_pts = self.mc.scatter([], [], s=16, color=P.ORANGE, lw=0, zorder=3)
        self.mc_note = self.mc.text(0.04, 0.96, "", transform=self.mc.transAxes, fontsize=11,
                                    color=P.INK2, va="top")
        self.mc.set_visible(False)

        self.Normalize, self.PowerNorm = Normalize, PowerNorm

    # ---------------------------------------------------------- setters
    def narrate(self, step="", title="", body="", eq="", alpha=1.0, width=46):
        self.step.set_text(step)
        self.title.set_text(title)
        self.body.set_text("\n\n".join(textwrap.fill(par, width) for par in body.split("\n\n")) if body else "")
        self.eq.set_text(eq)
        for t in self.texts[:4]:
            t.set_alpha(alpha)
        self.col_a.set_text("")
        self.col_b.set_text("")

    def show_cbar(self, cmap, label="", hi="1"):
        on = cmap is not None
        for a in (self.cax, ):
            a.set_visible(on)
        for t in (self.cbar_lo, self.cbar_hi, self.cbar_label):
            t.set_visible(on)
        if on:
            self.cbar.set_cmap(cmap)
            self.cbar_label.set_text(label)
            self.cbar_hi.set_text(hi)

    def peg_colors(self, values, cmap, norm, reveal=None):
        """RGBA of the pegs for values (NP,), blended from the bare peg by reveal (NP,) in [0, 1]."""
        base = np.tile(to_rgba(PEG_BASE), (NP, 1))
        col = cmap(norm(np.clip(values, 0, None)))
        if reveal is None:
            reveal = np.ones(NP)
        return base + (col - base) * np.clip(reveal, 0, 1)[:, None]

    def set_pegs(self, rgba):
        self.pegs.set_facecolors(rgba)

    def set_glyphs(self, p, alpha=1.0):
        """Two spokes per peg, down-left and down-right, as long as their probabilities."""
        if p is None:
            self.glyphs.set_segments([])
            return
        d = np.array([[-0.5, -DY], [0.5, -DY]])
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        pr = flat(p)
        segs = []
        for i in range(NP):
            for side, q in ((0, 1 - pr[i]), (1, pr[i])):
                segs.append([PXY[i], PXY[i] + 0.7 * q * d[side]])
        self.glyphs.set_segments(segs)
        self.glyphs.set_alpha(alpha)

    def set_flow(self, f, fmax, reveal=None, art=None, alpha=1.0):
        art = art or self.flow
        w = 11.0 * (np.clip(f, 0, None) / fmax) ** 0.6
        if reveal is not None:
            w = w * np.clip(reveal, 0, 1)
        art.set_linewidths(w)
        art.set_alpha(alpha)

    def set_bars(self, heights, colors=None):
        for j, b in enumerate(self.bars):
            b.set_height(max(float(heights[j]), 0.0))
            if colors is not None:
                b.set_facecolor(colors[j] if not isinstance(colors, str) else colors)

    def set_exact(self, heights):
        if heights is None:
            self.exact.set_segments([])
            return
        self.exact.set_segments([[(bin_x(j) - 0.42, Y_FLOOR + h), (bin_x(j) + 0.42, Y_FLOOR + h)]
                                 for j, h in enumerate(heights) if h > 1e-3])

    def set_balls(self, xy, colors=None):
        self.balls.set_offsets(xy if len(xy) else np.empty((0, 2)))
        if colors is not None and len(xy):
            self.balls.set_facecolors(colors)

    def show_cup(self, on, alpha=1.0):
        self.cup.set_visible(on)
        self.cup_label.set_visible(on)
        self.cup.set_alpha(alpha)
        self.cup_label.set_alpha(alpha)

    def show_minis(self, on):
        for a, _, _ in self.minis:
            a.set_visible(on)
        for t in self.ops:
            t.set_visible(on)


def ball_positions(cols, t_rows):
    """xy of balls with columns cols (n, R + 1) at time t_rows (n,), in rows since launch.

    Each ball falls from the drop point onto peg (0, 0), hops from peg to peg with a small bounce,
    and leaves the last row for its bin.  Balls with t_rows outside [0, R + 1) are not drawn.
    """
    n = len(cols)
    seg = np.floor(t_rows).astype(int)
    s = t_rows - seg
    keep = (t_rows >= 0) & (seg <= R)
    pts = np.empty((n, R + 2, 2))
    pts[:, 0] = DROP
    pts[:, 1:R + 1] = peg_xy(np.arange(R)[None, :], cols[:, :R]) + [0, 0.3]
    pts[:, R + 1, 0] = bin_x(cols[:, R])
    pts[:, R + 1, 1] = Y_BIN_TOP - 0.6
    seg_c = np.clip(seg, 0, R)
    a = pts[np.arange(n), seg_c]
    b = pts[np.arange(n), seg_c + 1]
    xy = a + (b - a) * np.stack([s, s ** 1.6], axis=1)
    hop = (seg_c >= 1) & (seg_c < R)
    xy[:, 1] += np.where(hop, 0.32 * np.sin(np.pi * s), 0.0)
    return xy, keep


# ------------------------------------------------------------------ scenes
class Video:
    def __init__(self, fps=24, n_balls=(450, 600, 4000), seed=1):
        self.fps = fps
        self.st = Stage()
        self.rng = np.random.default_rng(seed)
        self.n_fair, self.n_lean, self.n_check = n_balls
        self.p0, self.p1 = fair_board(), leaning_board()
        self.S0, self.bins0 = forward(self.p0)
        self.S1, self.bins1 = forward(self.p1)
        self.beta = backward(self.p1, TARGET)
        self.rho = self.S1 * self.beta
        self.f0 = edge_flow(self.p0, self.S0)
        self.f1 = edge_flow(self.p1, self.S1)
        self.fb = edge_flow(self.p1, self.S1, self.beta, TARGET)
        self.fmax = 0.5
        # the checks a viewer cannot see: the two passes agree with each other
        assert np.isclose(self.bins1[TARGET].sum(), self.beta[0, 0])
        assert np.allclose(self.rho.sum(axis=1)[:R], self.beta[0, 0])        # every row: P(cup)
        self.scale0 = 0.85 * H_BINS / self.bins0.max()
        self.scale1 = 0.85 * H_BINS / self.bins1.max()
        S = self.st
        self.mass, self.prob = P.mass_cmap(), P.prob_cmap()
        self.norm_s = S.PowerNorm(0.5, 0, 1)
        self.norm_b = S.Normalize(0, 1)
        self.p_cup = self.beta[0, 0]
        self.norm_r = S.PowerNorm(0.5, 0, self.p_cup)

    def sec(self, s):
        return max(1, int(round(s * self.fps)))

    # ---------------------------------------------------------- helpers
    def drop(self, p, n, seconds, frames_per_row, scale, narrate, colorize=None, on_land=None):
        """Animate n balls falling through board p; bars count arrivals (heights = scale * count / n)."""
        st = self.st
        cols = simulate(p, n, self.rng)
        total = self.sec(seconds)
        launch_span = total - (R + 2) * frames_per_row - self.sec(0.5)
        launch = launch_span * (np.arange(n) / n) ** 0.6
        land = launch + (R + 1) * frames_per_row
        counts = np.zeros(R + 1)
        landed = 0
        for f in range(total):
            narrate(f)
            t = (f - launch) / frames_per_row
            xy, keep = ball_positions(cols, t)
            now = land <= f
            for i in range(landed, now.sum()):
                counts[cols[i, R]] += 1
                if on_land is not None:
                    on_land(cols[i])
            landed = int(now.sum())
            colors = colorize(cols[keep], t[keep]) if colorize is not None else None
            st.set_balls(xy[keep], colors)
            st.set_bars(scale * counts / n)
            yield f, counts, landed
        st.set_balls([])

    # ---------------------------------------------------------- 1 title
    def s_title(self):
        st = self.st
        st.set_glyphs(None)
        st.set_pegs(st.peg_colors(np.zeros(NP), self.mass, self.norm_s, np.zeros(NP)))
        n = self.sec(5)
        for f in range(n):
            a = ease(f / self.sec(0.8))
            st.narrate("MPAT, EXPLAINED", "Plinko, Markov chains\nand bundles",
                       "Probabilistic tractography drops streamlines into the brain and lets each one "
                       "bounce from voxel to voxel at random. MPAT computes where all of them go, "
                       "exactly, without dropping a single one.\n\nThis video builds the idea on a "
                       "Plinko board.", alpha=a)
            st.body.set_y(0.74)
            yield

    # ---------------------------------------------------------- 2 fair board: Monte Carlo
    def s_fair_drop(self):
        st = self.st
        st.body.set_y(0.8)
        st.set_glyphs(self.p0)

        def narrate(f):
            st.narrate("1 · A FAIR BOARD", "Balls make a bell curve",
                       "At every peg a ball bounces left or right with equal probability, ½. Its "
                       "bin is the sum of 12 coin flips.\n\nDrop enough balls and the bins fill in a "
                       "bell curve: the binomial distribution, which is very close to a Gaussian.",
                       alpha=ease(f / self.sec(0.6)))
            st.eq.set_text(r"$P(\mathrm{bin}\ j) = \binom{12}{j}\,/\,2^{12}$" if f > self.sec(6) else "")

        for _ in self.drop(self.p0, self.n_fair, 13, 3, self.scale0, narrate):
            yield
        # the Gaussian
        x = np.linspace(bin_x(0) - 0.5, bin_x(R) + 0.5, 300)
        var = R / 4
        y = Y_FLOOR + self.scale0 * np.exp(-x ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)
        n = self.sec(2.5)
        for f in range(n):
            m = int(len(x) * ease(f / self.sec(1.2)))
            st.curve.set_data(x[:m], y[:m])
            yield

    # ---------------------------------------------------------- 3 fair board: forward pass
    def s_fair_exact(self):
        st = self.st
        st.show_cbar(self.mass, "P(ball passes this peg)")
        n_anim = self.sec(6)
        bars_mc = np.array([b.get_height() for b in st.bars])
        s0 = flat(self.S0)
        for f in range(n_anim + self.sec(3.5)):
            r = (R + 1) * ease(f / n_anim) if f < n_anim else R + 1
            st.narrate("1 · A FAIR BOARD", "…or push the probability",
                       "We don't need balls. Start with all the probability on the top peg and "
                       "push it down one row at a time: each peg sends ½ of what it holds to each "
                       "side.\n\nTwelve steps give the exact distribution (Pascal's triangle), the "
                       "curve the balls were scattering around.",
                       r"$x_{t+1} = Q^{\top} x_t$", alpha=ease(f / self.sec(0.6)))
            st.set_pegs(st.peg_colors(s0, self.mass, self.norm_s, r - PK))
            st.set_flow(self.f0, self.fmax, r - E_ROW - 0.3)
            st.glyphs.set_alpha(1 - 0.7 * ease(r / 3))
            st.set_bars(lerp(bars_mc, self.scale0 * self.bins0, r - R))
            if r > R:
                st.set_exact(self.scale0 * self.bins0)
            yield

    # ---------------------------------------------------------- 4 leaning board
    def s_lean(self):
        st = self.st
        st.curve.set_data([], [])
        st.set_exact(None)
        bars = np.array([b.get_height() for b in st.bars])
        s0 = flat(self.S0)
        n = self.sec(2.5)
        for f in range(n + self.sec(5.5)):
            a = ease(f / n)
            st.narrate("2 · A LEANING BOARD", "Every peg leans its own way",
                       "Now each peg has its own probability p of sending a ball right. The spokes "
                       "under each peg are as long as the two probabilities.\n\nWhere a ball goes next depends only on the peg it is on, "
                       "not on how it got there: a Markov chain. MPAT's tracking is one too: its pegs "
                       "are a voxel plus the walker's heading, and their lean comes from the fODF.",
                       r"$Q[\mathrm{peg} \to \mathrm{right}] = p_{\mathrm{peg}}$",
                       alpha=ease(f / self.sec(0.6)))
            st.set_glyphs(self.p0 + (self.p1 - self.p0) * a, 0.3 + 0.7 * a)
            st.set_pegs(st.peg_colors(s0, self.mass, self.norm_s, 1 - a * np.ones(NP)))
            st.set_flow(self.f0, self.fmax, (1 - a) * np.ones(len(EDGES)))
            st.set_bars(bars * (1 - a))
            st.show_cbar(self.mass if a < 0.5 else None, "P(ball passes this peg)")
            yield

    # ---------------------------------------------------------- 5 leaning board: drop + forward
    def s_lean_drop(self):
        st = self.st

        def narrate(f):
            st.narrate("2 · A LEANING BOARD", "A different distribution",
                       "The top pegs push balls away from the middle, and lower down a current "
                       "carries them right. The balls no longer make a bell curve: they split into "
                       "two streams.", alpha=1.0)

        for _ in self.drop(self.p1, self.n_lean, 13, 3, self.scale1, narrate):
            yield
        st.show_cbar(self.mass, "S: P(ball passes this peg)")
        bars_mc = np.array([b.get_height() for b in st.bars])
        s1 = flat(self.S1)
        n_anim = self.sec(5)
        for f in range(n_anim + self.sec(4)):
            r = (R + 1) * ease(f / n_anim) if f < n_anim else R + 1
            st.narrate("2 · A LEANING BOARD", "The forward pass",
                       "The same push, row by row, now with each peg's own p, gives the exact "
                       "distribution again, and also S: the probability that a ball passes each "
                       "peg (colour), and how many balls cross each gap (width).\n\nThis is MPAT's "
                       "forward pass: products with Qᵀ from the seeds, until the walkers are absorbed.",
                       r"$S = \sum_{t} (Q^{\top})^t \pi_0 = (I - Q^{\top})^{-1}\pi_0$",
                       alpha=ease(f / self.sec(0.6)))
            st.set_pegs(st.peg_colors(s1, self.mass, self.norm_s, r - PK))
            st.set_flow(self.f1, self.fmax, r - E_ROW - 0.3)
            st.glyphs.set_alpha(1 - 0.7 * ease(r / 3))
            st.set_bars(lerp(bars_mc, self.scale1 * self.bins1, r - R))
            if r > R:
                st.set_exact(self.scale1 * self.bins1)
            yield

    # ---------------------------------------------------------- 6 the bundle question
    def s_question(self):
        st = self.st
        st.set_exact(None)
        n = self.sec(9)
        for f in range(n):
            a = ease(f / self.sec(1.0))
            st.narrate("3 · A BUNDLE", "Which balls land in the cup?",
                       "A bundle is a question about where streamlines end: only the balls that land "
                       "in the target cup (the bundle's end ROI) belong to it.\n\nThe forward map "
                       "can't answer it. S counts every ball that passes a peg, including the ones "
                       "that go on to miss. Even next to the cup, many balls end up elsewhere.",
                       alpha=ease(f / self.sec(0.6)))
            st.show_cup(True, a)
            st.cup_fill.set_alpha(0.12 * a)
            grey = tuple(lerp(np.array(to_rgba(P.MUTED)), np.array(to_rgba(MISS)), a))
            st.set_bars(self.scale1 * self.bins1, [CUP if t else grey for t in TARGET])
            yield

    # ---------------------------------------------------------- 7 backward pass
    def s_backward(self):
        st = self.st
        s1, b1 = flat(self.S1), flat(self.beta)
        n_fade = self.sec(1.5)
        n_anim = self.sec(8)
        for f in range(n_fade + n_anim + self.sec(4.5)):
            g = f - n_fade
            r = (R + 1) * ease(g / n_anim) if g > 0 else 0.0
            st.narrate("4 · THE BACKWARD PASS", "Who makes it to the cup?",
                       "Start at the bottom: β = 1 in the cup's bins, 0 in the others. Then walk up "
                       "the board. A ball on a peg lands in the cup if it bounces right and makes it "
                       "from there, or bounces left and makes it from there.\n\nβ is the probability "
                       "that a ball on this peg ends in the cup. One sweep gives it for every peg at "
                       "once: products with Q instead of Qᵀ.",
                       r"$\beta_{\mathrm{peg}} = p\,\beta_{\mathrm{right}} + (1-p)\,\beta_{\mathrm{left}}$"
                       "\n\n" r"$\beta = \sum_t Q^t r = (I - Q)^{-1} r$",
                       alpha=ease(f / self.sec(0.6)))
            if f < n_fade:
                a = ease(f / n_fade)
                st.set_pegs(st.peg_colors(s1, self.mass, self.norm_s, (1 - a) * np.ones(NP)))
                st.set_flow(self.f1, self.fmax, (1 - a) * np.ones(len(EDGES)))
                st.glyphs.set_alpha(0.3 + 0.7 * a)
                st.show_cbar(None)
            else:
                st.show_cbar(self.prob, "β: P(ball on this peg ends in the cup)")
                st.set_pegs(st.peg_colors(b1, self.prob, self.norm_b, r - (R - PK)))
                st.beta_label.set_visible(True)
                st.beta_label.set_alpha(ease(g / self.sec(0.6)))
            yield
        st.beta_label.set_visible(False)

    # ---------------------------------------------------------- 8 the product
    def s_product(self):
        st = self.st
        s1, b1, rho = flat(self.S1), flat(self.beta), flat(self.rho)
        for (a, sc, _), v, cm, nm in zip(st.minis, (s1, b1, rho), (self.mass, self.prob, self.mass),
                                         (self.norm_s, self.norm_b, self.norm_r)):
            sc.set_facecolors(cm(nm(v)))
        st.show_minis(True)
        n = self.sec(3)
        bars_all = self.scale1 * self.bins1
        bars_cup = np.where(TARGET, bars_all, 0)
        for f in range(n + self.sec(10)):
            a = ease(f / n)
            st.narrate("5 · OCCUPANCY × COMPLETION", "Multiply the two passes",
                       "A visit to a peg belongs to the bundle when the ball got there (S) and will "
                       "go on to land in the cup (β). Given the peg, the past and the future are "
                       "independent (Markov), so the bundle's visits are just S × β.\n\nThe second "
                       "stream, and every ball that misses, is gone. This is MPAT's bundle density.",
                       r"$\rho = S \cdot \beta$", alpha=ease(f / self.sec(0.6)))
            st.eq.set_y(0.42)
            cb = st.peg_colors(b1, self.prob, self.norm_b)
            cr = st.peg_colors(rho, self.mass, self.norm_r)
            st.set_pegs(cb + (cr - cb) * a)
            st.glyphs.set_alpha(1 - 0.75 * a)
            st.set_flow(self.f1, self.fmax, a * np.ones(len(EDGES)), art=st.ghost, alpha=0.8)
            st.set_flow(self.fb, self.fmax, a * np.ones(len(EDGES)))
            st.set_bars(lerp(bars_all, bars_cup, a))
            if a < 0.5:
                st.show_cbar(self.prob, "β: P(ball on this peg ends in the cup)")
            else:
                st.show_cbar(self.mass, "ρ: P(ball passes this peg and ends in the cup)",
                             f"{self.p_cup:.2f}")
            yield
        st.eq.set_y(0.44)

    # ---------------------------------------------------------- 9 check, then the dictionary
    def s_check(self):
        st = self.st
        st.show_minis(False)
        st.mc.set_visible(True)
        rho = flat(self.rho)
        hi = rho.max() * 1.08
        st.mc.plot([0, hi], [0, hi], color=P.MUTED, lw=1.2, zorder=1)
        st.mc.set_xlim(0, hi)
        st.mc.set_ylim(0, hi)
        hits = np.zeros(NP)
        idx = {pk: i for i, pk in enumerate(PEGS)}
        st.set_exact(np.where(TARGET, self.scale1 * self.bins1, 0))
        miss_rgba, ball_rgba, cup_rgba = (np.array(to_rgba(c)) for c in (MISS, BALL, CUP))

        def on_land(c):
            if TARGET[c[R]]:
                for k in range(R):
                    hits[idx[(k, c[k])]] += 1

        def colorize(cols, t):
            out = np.tile(ball_rgba, (len(cols), 1))
            fate = TARGET[cols[:, R]]
            late = t >= R
            out[late & fate] = cup_rgba
            out[late & ~fate] = miss_rgba
            return out

        state = {"n": 0}

        def narrate(f):
            st.narrate("6 · THE CHECK", "Keep only the hits",
                       "Drop balls and keep only those that land in the cup. Their visits to each "
                       "peg converge to ρ, with an error that shrinks only as 1/√N.\n\nMPAT gets ρ "
                       "exactly, with no balls. In the brain, that is the bundle density of "
                       "infinitely many streamlines.", alpha=ease(f / self.sec(0.6)))
            n = max(state["n"], 1)
            st.mc_pts.set_offsets(np.c_[rho, hits / n])
            err = np.abs(hits / n - rho).max()
            st.mc_note.set_text(f"{state['n']:,} balls\nmax error {err:.3f}" if state["n"] else "")

        gen = self.drop(self.p1, self.n_check, 14, 2, self.scale1, narrate, colorize, on_land)
        # bars: only the cup's bins count; misses drop through
        for f, counts, landed in gen:
            state["n"] = landed
            st.set_bars(np.where(TARGET, self.scale1 * counts / self.n_check, 0),
                        [CUP if t else P.MUTED for t in TARGET])
            yield
        for _ in range(self.sec(2.5)):
            yield

    def s_dictionary(self):
        st = self.st
        st.mc.set_visible(False)
        st.set_exact(None)
        rows = [("Plinko board", "MPAT"),
                ("peg", "state: (WM voxel, heading)"),
                ("a peg's lean p", "transitions Q (fODF, MITTENS)"),
                ("bins", "absorbing: grey matter, edge"),
                ("drop point", "seeds: launches π₀"),
                ("target cup", "end ROI: r"),
                ("forward push", "S = (I − Qᵀ)⁻¹ π₀"),
                ("backward sweep", "β = (I − Q)⁻¹ r"),
                ("S × β", "bundle density ρ")]
        n = self.sec(14)
        for f in range(n):
            st.narrate("SUMMARY", "From the board to the brain",
                       alpha=ease(f / self.sec(0.6)))
            k = min(len(rows), 1 + int(f / self.sec(0.7)))
            st.col_a.set_text("\n".join(r[0] for r in rows[:k]))
            st.col_b.set_text("\n".join(r[1] for r in rows[:k]))
            st.col_a.set_alpha(1)
            st.col_b.set_alpha(1)
            st.body.set_y(0.27)
            if f > self.sec(6.5):
                st.body.set_text(textwrap.fill(
                    "On the board every ball takes exactly 12 steps. Walkers in the brain can wander "
                    "for hundreds, so MPAT sums each series until the mass still moving falls below "
                    "tol. Include and exclude ROIs, and the two halves of every streamline, are more "
                    "forward and backward columns through the same products.", 50))
                st.body.set_alpha(ease((f - self.sec(6.5)) / self.sec(0.6)))
                st.body.set_fontsize(12.5)
            yield

    def scenes(self):
        return [self.s_title, self.s_fair_drop, self.s_fair_exact, self.s_lean, self.s_lean_drop,
                self.s_question, self.s_backward, self.s_product, self.s_check, self.s_dictionary]


# ------------------------------------------------------------------ rendering
def render(out, fps=24, gif=False, preview=False, quick=False):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.animation import FFMpegWriter, PillowWriter

    n_balls = (250, 300, 1500) if quick else (450, 600, 4000)
    v = Video(fps=fps, n_balls=n_balls)
    fig = v.st.fig
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if preview:  # a still every two seconds and at the end of every scene
        pdir = out.parent / "preview"
        pdir.mkdir(exist_ok=True)
        total = 0
        for n, scene in enumerate(v.scenes()):
            name = scene.__name__[2:]
            i = 0
            for i, _ in enumerate(scene()):
                if i and i % (2 * fps) == 0:
                    fig.savefig(pdir / f"{n:02d}_{name}_{i:04d}.png")
            fig.savefig(pdir / f"{n:02d}_{name}_end.png")
            total += i + 1
            print(f"scene {n} {name}: {i + 1} frames ({(i + 1) / fps:.1f} s)")
        print(f"total {total / fps:.0f} s")
        return

    if gif:
        writer = PillowWriter(fps=fps)
        out = out.with_suffix(".gif")
    else:
        import imageio_ffmpeg

        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=-1,
                              extra_args=["-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart"])
    n = 0
    with writer.saving(fig, str(out), dpi=100):
        for scene in v.scenes():
            for _ in scene():
                writer.grab_frame()
                n += 1
    print(f"wrote {out} ({n} frames, {n / fps:.0f} s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-o", "--out", default=str(HERE / "output" / "mpat_plinko.mp4"))
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--gif", action="store_true", help="write an animated GIF (no ffmpeg needed)")
    ap.add_argument("--preview", action="store_true", help="write stills of every scene instead")
    ap.add_argument("--quick", action="store_true", help="fewer balls, 12 fps: a fast draft")
    a = ap.parse_args()
    render(a.out, fps=12 if a.quick else a.fps, gif=a.gif, preview=a.preview, quick=a.quick)
