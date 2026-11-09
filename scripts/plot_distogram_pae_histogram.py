"""Histogram of the pAE values a mosaic_scan --distogram_loss_weights run actually achieved, for
the two interface terms _passing() gates on (binder_target_pae, target_binder_pae) plus
within_binder_pae for context. One figure, three panels, all weights/targets pooled.

Usage:
    python3 scripts/plot_distogram_pae_histogram.py --run runs/mosaic/mosaic_distogram_sweep_af2 --label af2
"""

import os
import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, GREY, INK = "#4C72B0", "#8C8C8C", "#2B2B2B"
PANELS = [
    ("within_binder_pae", "within-binder pAE (Å)", None),
    ("binder_target_pae", "binder→target pAE (Å)", 5.0),
    ("target_binder_pae", "target→binder pAE (Å)", 5.0),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="sweep run directory (has results.csv)")
    ap.add_argument("--label", required=True, help="short name used in the title/filename, e.g. af2")
    ap.add_argument("--stage", default="stage_2", help="stage to plot (default: stage_2)")
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

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    for ax, (col, label, threshold) in zip(axes, PANELS):
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        ax.hist(s, bins=24, color=BLUE, edgecolor="white", linewidth=0.6, zorder=3)
        ax.axvline(s.median(), color=INK, lw=1.6, zorder=4)
        ax.text(s.median(), ax.get_ylim()[1], f" median {s.median():.1f}", rotation=90,
               va="top", ha="left", fontsize=8.5, color=INK)
        if threshold is not None:
            ax.axvline(threshold, color=GREY, ls="--", lw=1.3, zorder=4)
            frac = (s < threshold).mean()
            ax.text(threshold, ax.get_ylim()[1], f" <{threshold:g} Å: {frac*100:.0f}% ",
                   rotation=90, va="top", ha="right", fontsize=8.5, color="#555555")
        ax.set_xlabel(label, fontsize=10.5)
        ax.grid(axis="y", color="#E6E6E6", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("designs", fontsize=10.5)
    n = len(df)
    fig.suptitle(f"{args.label}: achieved pAE, {args.stage}  (n={n}, all weights/targets pooled)",
                fontsize=13, fontweight="bold", y=1.01)
    fig.text(0.5, -0.04, "Dashed line = the _passing() threshold (binder→target / "
                        "target→binder only); within-binder pAE has no gate.",
             ha="center", va="bottom", fontsize=8.5, color="#666666")
    fig.tight_layout()
    path = os.path.join(args.out_path, f"pae_histogram_{args.label}.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"figure: {path}")


if __name__ == "__main__":
    main()
