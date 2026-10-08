"""Plot a landscape.py distogram_loss_weight sweep, in the style of the landscape/mosaic_scan
run figures (see scripts/plot_mosaic_scan.py).

Produces one figure next to the run's results.csv:

  landscape_distogram_sweep_metrics.png   metric distributions, split by distogram_loss_weight
                                           (one KDE per weight, all metrics in one grid)

Usage:
    python3 scripts/plot_landscape_distogram_sweep.py --run_path runs/landscape_distogram_sweep/
"""

import os
import argparse

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

GREY = "#8C8C8C"
# House palette from the landscape/mosaic_scan figures (seaborn "deep"-like) -- clearly
# distinct hues for a small categorical subset, as opposed to the sequential viridis used
# for the full sweep. PALETTE extends it far enough for any subset figure; BLUE/ORANGE/GREEN
# are kept as names since earlier subset figures already reference them directly.
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860", "#DA8BC3"]
BLUE, ORANGE, GREEN = PALETTE[0], PALETTE[1], PALETTE[2]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "font.size": 10,
})

# (column, label) -- every numeric per-design metric in landscape.py's results.csv, excluding
# the sweep variable itself (distogram_loss_weight) and string/id columns (attempt, stage,
# init, sequence, dssp_salad, dssp_predicted).
METRICS = [
    ("plddt",             "pLDDT"),
    ("ptm",               "pTM"),
    ("recovery",          "Self-consistency recovery"),
    ("entropy",           "Self-consistency entropy"),
    ("contacts",          "Contacts\n(entropy, lower = tighter)"),
    ("pae",               "pAE"),
    ("distogram_nll",     "Distogram NLL vs blueprint"),
    ("rmsd_to_blueprint", "CA-RMSD to blueprint (Å)"),
    ("dssp_identity",     "DSSP identity vs blueprint"),
    ("average_num_nonzero", "Avg. # nonzero AAs (soft seq)"),
    ("L",                 "DSSP loop fraction"),
    ("H",                 "DSSP helix fraction"),
    ("E",                 "DSSP strand fraction"),
]


def load(run_path):
    return pd.read_csv(os.path.join(run_path, "results.csv"))


def _weights_matching(all_weights, targets, atol=1e-3):
    """Pick the sweep's actual unique weight values closest to `targets` (values round-trip
    through CSV as e.g. np.float32(0.1) -> 0.1000000014901161, so exact equality would miss)."""
    return [w for w in all_weights if any(np.isclose(w, t, atol=atol) for t in targets)]


def _kde(ax, values, color, label, fill=True):
    """Silverman-bandwidth KDE, with a fallback for degenerate data (matches
    scripts/plot_mosaic_scan.py's _kde)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return
    if v.size < 3 or np.isclose(v.std(), 0.0):
        ax.axvline(v.mean(), color=color, lw=1.8, label=label)
        return
    pad = 0.05 * (v.max() - v.min() + 1e-9)
    grid = np.linspace(v.min() - pad, v.max() + pad, 512)
    dens = gaussian_kde(v, bw_method="silverman")(grid)
    ax.plot(grid, dens, color=color, lw=1.8, label=label)
    if fill:
        ax.fill_between(grid, dens, color=color, alpha=0.15)


def _dot_legend(fig, entries, y=0.945):
    handles = [Line2D([], [], marker="o", ls="none", ms=9, color=c, label=l)
               for l, c in entries]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y),
               ncol=len(entries), frameon=False, fontsize=11, handletextpad=0.4,
               columnspacing=2.2)


def _footnote(fig, text, y=0.012):
    fig.text(0.5, y, text, ha="center", va="bottom", fontsize=8.5, color="#666666")


# ---------------------------------------------------------------------------------------
# metric distributions, one KDE per distogram_loss_weight
# ---------------------------------------------------------------------------------------

def fig_metrics(results, out_path, run_name, weights=None, colors=None, suffix=""):
    """One KDE-grid figure across all metrics, one curve per `weights` value (defaults to
    every weight in the sweep, colored with a sequential colormap since the sweep variable
    is ordered). Pass an explicit `weights` subset + `colors` dict (value -> hex/RGBA) to
    restrict/recolor the comparison, e.g. a clearly-distinct 2-color pair."""
    all_weights = sorted(results["distogram_loss_weight"].unique())
    if weights is None:
        weights = all_weights
    if colors is None:
        # Sequential colormap: the sweep variable is ordered (0.0 -> 2.0), not categorical.
        cmap = plt.get_cmap("viridis", len(all_weights))
        colors = {w: cmap(i) for i, w in enumerate(all_weights)}
    label = {w: f"w = {w:g}" for w in weights}
    sub = results[results["distogram_loss_weight"].isin(weights)]

    fig, axes = plt.subplots(4, 4, figsize=(19, 15.0))
    for ax, (col, lab) in zip(axes.ravel(), METRICS):
        for w in weights:
            _kde(ax, sub.loc[sub["distogram_loss_weight"] == w, col], colors[w], label[w])
        ax.set_title(lab)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    for ax in axes.ravel()[len(METRICS):]:
        ax.set_visible(False)

    subset_note = "" if len(weights) == len(all_weights) else f" ({', '.join(label[w] for w in weights)} only)"
    fig.suptitle("landscape distogram-weight sweep — metric distributions by "
                 f"distogram_loss_weight{subset_note}   ({run_name})", fontsize=15, y=0.99)
    _dot_legend(fig, [(label[w], colors[w]) for w in weights], y=0.955)
    n_a = sub["attempt"].nunique()
    _footnote(fig, f"n = {len(sub)} rows ({n_a} blueprints × {len(weights)} weight"
                   f"{'s' if len(weights) != 1 else ''}; the same salad blueprint is reused "
                   "across weights for each attempt). KDE, Silverman bandwidth.")
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    path = os.path.join(out_path, f"landscape_distogram_sweep_metrics{suffix}.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_path", default="runs/landscape_distogram_sweep/")
    ap.add_argument("--out_path", default=None, help="defaults to --run_path")
    args = ap.parse_args()
    out_path = args.out_path or args.run_path
    os.makedirs(out_path, exist_ok=True)
    run_name = os.path.basename(os.path.normpath(args.run_path))

    results = load(args.run_path)
    print(f"loaded {len(results)} rows, {results['attempt'].nunique()} attempts, "
          f"{results['distogram_loss_weight'].nunique()} weights: "
          f"{sorted(results['distogram_loss_weight'].unique())}")

    paths = [fig_metrics(results, out_path, run_name)]

    # Baseline (lowest weight, normally 0.0) vs the next 1 and next 2 weights up the
    # sweep -- adapts to whatever weights this run actually used, rather than hardcoding
    # literal values that only matched one particular sweep's spacing.
    all_weights = sorted(results["distogram_loss_weight"].unique())

    def _subset_figure(weights, palette, suffix):
        paths.append(fig_metrics(results, out_path, run_name,
                                 weights=weights,
                                 colors=dict(zip(weights, palette)),
                                 suffix=suffix))

    if len(all_weights) >= 2:
        pair = all_weights[:2]
        _subset_figure(pair, [BLUE, ORANGE],
                       f"_w{pair[0]:g}_vs_w{pair[1]:g}")
    if len(all_weights) >= 3:
        triple = all_weights[:3]
        _subset_figure(triple, [BLUE, ORANGE, GREEN],
                       f"_w{triple[0]:g}_vs_w{triple[1]:g}_vs_w{triple[2]:g}")

    # Explicit fixed comparison (0.0 vs 0.1) regardless of how the sweep is spaced --
    # distinct from the "pair" figure above, which is the two *smallest* weights and may
    # not be 0.1 (e.g. sweep2's smallest non-zero weight is 0.05). No-op if either is absent.
    explicit_pair = _weights_matching(all_weights, [0.0, 0.1])
    if len(explicit_pair) == 2 and explicit_pair != all_weights[:2]:
        _subset_figure(explicit_pair, [BLUE, ORANGE],
                       f"_w{explicit_pair[0]:g}_vs_w{explicit_pair[1]:g}")

    # Explicitly requested: every weight except 0.05 (still includes 0.1). No-op if this
    # sweep never had a 0.05 weight in the first place.
    no_005 = [w for w in all_weights if not np.isclose(w, 0.05, atol=1e-3)]
    if 2 <= len(no_005) < len(all_weights):
        if len(no_005) <= len(PALETTE):
            _subset_figure(no_005, PALETTE[:len(no_005)],
                           "_" + "_vs_".join(f"w{w:g}" for w in no_005))
        else:
            print(f"skipping w!=0.05 figure: {len(no_005)} weights but only "
                  f"{len(PALETTE)} palette colors defined")

    for p in paths:
        print("wrote", p)


if __name__ == "__main__":
    main()
