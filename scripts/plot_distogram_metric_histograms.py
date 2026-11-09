"""Histogram grid of every numeric metric in a mosaic_scan results.csv -- a full diagnostic dump,
one panel per column, rather than the curated subsets the other plot_distogram_*.py scripts pick.
All weights/targets pooled for the given stage.

Numeric columns are auto-detected (>90% of values parse as a number), so this works unchanged on
any run's schema (coldspot columns only present with --coldspot_file, Protenix's structure_model
column, etc.). target/attempt/stage/distogram_weight/target_length are experiment-design/grouping
variables, not measured outcomes, and are always excluded.

Usage:
    python3 scripts/plot_distogram_metric_histograms.py --run runs/mosaic/mosaic_distogram_sweep_af2 --label af2
"""

import os
import math
import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, INK = "#4C72B0", "#2B2B2B"
EXCLUDE = {"target", "attempt", "stage", "distogram_weight", "target_length"}


def numeric_columns(df):
    cols = []
    for c in df.columns:
        if c in EXCLUDE:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() > 0.9:  # mostly real numbers, not a string column or a "none"-filled
            cols.append(c)         # one absent in this run's configuration (e.g. no hotspot_file)
    return cols


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="sweep run directory (has results.csv)")
    ap.add_argument("--label", required=True, help="short name used in the title/filename, e.g. af2")
    ap.add_argument("--stage", default="stage_2", help="stage to plot (default: stage_2)")
    ap.add_argument("--ncols", type=int, default=6, help="panel grid width (default: 6)")
    ap.add_argument("--out_path", default="runs/mosaic/mosaic_distogram_analysis",
                    help="where to write the PNG")
    args = ap.parse_args()
    os.makedirs(args.out_path, exist_ok=True)

    fp = os.path.join(args.run, "results.csv")
    if not os.path.isfile(fp):
        raise SystemExit(f"no results.csv in {args.run}")
    df = pd.read_csv(fp)
    df = df.drop_duplicates(subset=["target", "attempt", "stage", "distogram_weight"], keep="last")
    df = df[df.stage == args.stage]
    if df.empty:
        raise SystemExit(f"no {args.stage} rows in {fp}")

    cols = numeric_columns(df)
    if not cols:
        raise SystemExit("no numeric metric columns found")
    nrows = math.ceil(len(cols) / args.ncols)
    fig, axes = plt.subplots(nrows, args.ncols, figsize=(3.1 * args.ncols, 2.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, cols):
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        nbins = int(np.clip(s.nunique(), 5, 20))
        ax.hist(s, bins=nbins, color=BLUE, edgecolor="white", linewidth=0.5, zorder=3)
        ax.axvline(s.median(), color=INK, lw=1.3, zorder=4)
        ax.set_title(col.replace("_", " "), fontsize=9.5)
        ax.text(0.98, 0.95, f"med {s.median():.3g}", transform=ax.transAxes, ha="right", va="top",
               fontsize=7.5, color=INK)
        ax.tick_params(labelsize=7.5)
        ax.grid(axis="y", color="#E6E6E6", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[len(cols):]:
        ax.set_visible(False)

    n = len(df)
    fig.suptitle(f"{args.label}: all metrics, {args.stage}  (n={n}, all weights/targets pooled)",
                fontsize=14, fontweight="bold", y=1.005)
    fig.tight_layout()
    path = os.path.join(args.out_path, f"metric_histograms_{args.label}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{len(cols)} metrics, {nrows}x{args.ncols} grid")
    print(f"figure: {path}")


if __name__ == "__main__":
    main()
