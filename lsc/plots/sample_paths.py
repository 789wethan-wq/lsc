"""Sample-path figure sheet for visual sanity check (M1 acceptance)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lsc.dgp import (
    AR1StateDGP,
    BreakSpec,
    LocalLevelDGP,
    LocalLinearTrendDGP,
    MarkovSwitchingDGP,
    TimeVaryingVolDGP,
)

CATALOG = [
    ("local level, null", LocalLevelDGP(q=0.5, r=1.0)),
    ("local level, abrupt level break 1σ @250",
     LocalLevelDGP(q=0.5, r=1.0, breaks=[BreakSpec("level", 0.5, magnitude=1.0)])),
    ("local level, abrupt level break 3σ @250",
     LocalLevelDGP(q=0.5, r=1.0, breaks=[BreakSpec("level", 0.5, magnitude=3.0)])),
    ("local level, gradual ramp 3σ (half-life 25)",
     LocalLevelDGP(q=0.5, r=1.0, breaks=[BreakSpec("ramp", 0.5, magnitude=3.0, half_life=25)])),
    ("local level, variance break ×3 @250",
     LocalLevelDGP(q=0.5, r=1.0, breaks=[BreakSpec("variance", 0.5, vol_mult=3.0)])),
    ("local level, 3 breaks 1σ each",
     LocalLevelDGP(q=0.5, r=1.0, breaks=[BreakSpec("level", f, magnitude=1.0) for f in (0.25, 0.5, 0.75)])),
    ("local level, t5 obs noise", LocalLevelDGP(q=0.5, r=1.0, t_dof=5)),
    ("local linear trend, null", LocalLinearTrendDGP()),
    ("AR(1) state φ=0.95, level break 3σ @250",
     AR1StateDGP(phi=0.95, q=0.5, r=1.0, breaks=[BreakSpec("level", 0.5, magnitude=3.0)])),
    ("AR(1) state, nonlinear drift", AR1StateDGP(phi=0.95, q=0.5, r=1.0, drift_coef=0.1)),
    ("Markov switching 2-state (p=0.98)", MarkovSwitchingDGP(persistence=0.98)),
    ("time-varying vol, ×3 @250",
     TimeVaryingVolDGP(breaks=[BreakSpec("variance", 0.5, vol_mult=3.0)])),
]


def make_figure_sheet(path: str, T: int = 500, seed: int = 42) -> None:
    n = len(CATALOG)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3 * nrows))
    for ax, (title, dgp) in zip(axes.ravel(), CATALOG):
        s = dgp.sample(T, seed=seed)
        ax.plot(s.Y, lw=0.5, color="gray", alpha=0.8, label="Y")
        ax.plot(s.S_true, lw=1.2, color="C0", label="S_true")
        for bt in s.break_times:
            ax.axvline(bt, color="red", ls="--", lw=0.8, alpha=0.6)
        ax.set_title(title, fontsize=9)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "paper_assets/dgp_sample_paths.png"
    make_figure_sheet(out)
    print(f"wrote {out}")
