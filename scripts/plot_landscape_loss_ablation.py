"""Plot a landscape.py --loss_ablation run, in the style of the landscape/mosaic_scan run
figures (see scripts/plot_mosaic_scan.py, scripts/plot_landscape_distogram_sweep.py).

Unlike the plain distogram_loss_weight sweep, this run varies TWO things together --
which loss terms are active (`loss_config`) and the distogram weight -- so weight alone
isn't a valid grouping key (w=0.1 means something different under `full_with_distogram`
than under `distogram_recovery_plddt`). Each figure groups by the 5 actual
(loss_config, weight) combinations instead.

Produces one figure next to the run's results.csv:

  landscape_loss_ablation_metrics.png   metric distributions, one KDE per (loss_config, weight)

Usage:
    python3 scripts/plot_landscape_loss_ablation.py --run_path runs/landscape_loss_ablation/
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
# House palette from the landscape/mosaic_scan figures (seaborn "deep"-like).
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860", "#DA8BC3"]

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
# the two sweep variables (loss_config, distogram_loss_weight) and string/id columns
# (attempt, stage, init, sequence, dssp_salad, dssp_predicted).
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

# Preferred display order for the 3 loss configs (baseline first, then the two variants
# each ablation weight is tested under) -- falls back to sorted-by-appearance for any
# config name not listed here, so an unexpected label still gets plotted.
CONFIG_ORDER = ["no_distogram", "full_with_distogram", "distogram_recovery_plddt"]


def load(run_path):
    return pd.read_csv(os.path.join(run_path, "results.csv"))


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
               ncol=min(len(entries), 3), frameon=False, fontsize=10.5, handletextpad=0.4,
               columnspacing=2.0)


def _footnote(fig, text, y=0.012):
    fig.text(0.5, y, text, ha="center", va="bottom", fontsize=8.5, color="#666666")


def _runs(results):
    """The distinct (loss_config, weight) combinations actually present, ordered by
    CONFIG_ORDER then by weight, each with a fixed label and PALETTE color."""
    combos = results[["loss_config", "distogram_loss_weight"]].drop_duplicates()
    order_key = {c: i for i, c in enumerate(CONFIG_ORDER)}
    combos = combos.assign(_o=combos["loss_config"].map(order_key).fillna(len(CONFIG_ORDER)))
    combos = combos.sort_values(["_o", "distogram_loss_weight"])
    runs = []
    for _, row in combos.iterrows():
        cfg, w = row["loss_config"], row["distogram_loss_weight"]
        label = cfg if cfg == "no_distogram" else f"{cfg} (w={w:g})"
        runs.append((cfg, w, label))
    return runs


# ---------------------------------------------------------------------------------------
# metric distributions, one KDE per (loss_config, weight) combination
# ---------------------------------------------------------------------------------------

def fig_metrics(results, out_path, run_name, runs=None, colors=None, suffix=""):
    """One KDE-grid figure across all metrics, one curve per (loss_config, weight) in
    `runs` (defaults to every combination in the sweep). Pass an explicit `runs` subset to
    restrict the comparison -- `colors` (dict keyed by (loss_config, weight)) defaults to
    a fresh PALETTE assignment for that subset, but pass the *full* sweep's color map to
    keep e.g. no_distogram's color consistent with the all-combinations figure."""
    all_runs = _runs(results)
    if runs is None:
        runs = all_runs
    if colors is None:
        colors = dict(zip([(c, w) for c, w, _ in runs], PALETTE))
    labels = {(c, w): lab for c, w, lab in runs}

    fig, axes = plt.subplots(4, 4, figsize=(19, 15.0))
    for ax, (col, lab) in zip(axes.ravel(), METRICS):
        for cfg, w, _ in runs:
            sub = results[(results["loss_config"] == cfg) & (results["distogram_loss_weight"] == w)]
            _kde(ax, sub[col], colors[(cfg, w)], labels[(cfg, w)])
        ax.set_title(lab)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    for ax in axes.ravel()[len(METRICS):]:
        ax.set_visible(False)

    subset_note = "" if len(runs) == len(all_runs) else " (subset)"
    fig.suptitle(f"landscape loss ablation — metric distributions by loss_config × weight"
                 f"{subset_note}   ({run_name})", fontsize=15, y=0.99)
    _dot_legend(fig, [(labels[(c, w)], colors[(c, w)]) for c, w, _ in runs], y=0.955)
    wanted = {(c, w) for c, w, _ in runs}
    mask = pd.Series(list(zip(results["loss_config"], results["distogram_loss_weight"])),
                     index=results.index).isin(wanted)
    sub_all = results[mask]
    n_a = sub_all["attempt"].nunique()
    _footnote(fig, f"n = {len(sub_all)} rows ({n_a} blueprints × {len(runs)} (loss_config, weight) "
                   "combinations; the same salad blueprint is reused across all combinations "
                   "for each attempt). KDE, Silverman bandwidth.")
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    path = os.path.join(out_path, f"landscape_loss_ablation_metrics{suffix}.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_path", default="runs/landscape_loss_ablation/")
    ap.add_argument("--out_path", default=None, help="defaults to --run_path")
    args = ap.parse_args()
    out_path = args.out_path or args.run_path
    os.makedirs(out_path, exist_ok=True)
    run_name = os.path.basename(os.path.normpath(args.run_path))

    results = load(args.run_path)
    runs = _runs(results)
    print(f"loaded {len(results)} rows, {results['attempt'].nunique()} attempts, "
          f"{len(runs)} (loss_config, weight) combinations:")
    for cfg, w, lab in runs:
        print(f"  {lab}: {len(results[(results.loss_config == cfg) & (results.distogram_loss_weight == w)])} rows")

    colors = dict(zip([(c, w) for c, w, _ in runs], PALETTE))
    paths = [fig_metrics(results, out_path, run_name, runs=runs, colors=colors)]

    # Explicitly requested: no_distogram baseline together with just the ablation cases
    # (distogram_recovery_plddt), dropping full_with_distogram.
    no_distogram_and_ablation = [r for r in runs if r[0] in ("no_distogram", "distogram_recovery_plddt")]
    if 2 <= len(no_distogram_and_ablation) < len(runs):
        paths.append(fig_metrics(results, out_path, run_name,
                                 runs=no_distogram_and_ablation, colors=colors,
                                 suffix="_no_distogram_vs_ablation"))

    for p in paths:
        print("wrote", p)


if __name__ == "__main__":
    main()
