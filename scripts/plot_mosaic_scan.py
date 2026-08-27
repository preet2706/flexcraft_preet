"""Plot a mosaic_scan sweep, in the style of the landscape run figures.

Produces four figures next to the run's results.csv:

  mosaic_scan_metrics.png       metric distributions, stage_1 vs stage_2 (KDE grid)
  mosaic_scan_targets.png       per-target benchmark: yield + metric spread
  mosaic_scan_stage_scatter.png stage_1 vs stage_2 paired scatter (same blueprint)
  mosaic_scan_crossmetric.png   cross-metric relationships, split by pass/fail

Usage:
    python3 scripts/plot_mosaic_scan.py --run_path runs/mosaic_scan_L80/
"""

import os
import argparse

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, pearsonr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# House palette from the landscape figures (seaborn "deep"-like).
BLUE, ORANGE, GREEN, GREY = "#4C72B0", "#DD8452", "#55A868", "#8C8C8C"
STAGE_COLOR = {"stage_1": BLUE, "stage_2": ORANGE}
STAGE_LABEL = {"stage_1": "Stage 1 (soft APGM)", "stage_2": "Stage 2 (logit hardening)"}

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

# (column, label, pass-threshold, threshold direction) -- thresholds are the _passing()
# filter in the pipeline; direction is the side that PASSES.
METRICS = [
    ("plddt",                  "Binder pLDDT",                 0.80, "above"),
    ("ipsae",                  "ipSAE",                        0.75, "above"),
    ("iptm",                   "ipTM",                         None, None),
    ("recovery",               "Self-consistency recovery",    0.60, "above"),
    ("binder_target_pae",      "binder→target pAE (Å)",  5.0, "below"),
    ("target_binder_pae",      "target→binder pAE (Å)",  5.0, "below"),
    ("within_binder_pae",      "within-binder pAE (Å)",   None, None),
    ("entropy",                "Self-consistency entropy",     None, None),
    ("binder_target_contacts", "Binder–target contacts\n(entropy, lower = tighter)", None, None),
    ("within_binder_contacts", "Within-binder contacts\n(entropy, lower = tighter)",      None, None),
    ("rmsd_to_blueprint",      "CA-RMSD to blueprint (Å)", None, None),
    ("dssp_identity",          "DSSP identity vs blueprint",   None, None),
]


def load(run_path):
    results = pd.read_csv(os.path.join(run_path, "results.csv"))
    results["passed"] = results["passed"].astype(str).str.strip() == "True"
    success_path = os.path.join(run_path, "success.csv")
    success = pd.read_csv(success_path) if os.path.isfile(success_path) else None
    return results, success


def _kde(ax, values, color, label, fill=True):
    """Silverman-bandwidth KDE, with a histogram fallback for degenerate data."""
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


def _threshold(ax, thr, direction):
    """Dashed pass-threshold line, labelled near the axis so it never collides with the
    panel title."""
    if thr is None:
        return
    ax.axvline(thr, color=GREY, ls="--", lw=1.1, zorder=0)
    ax.text(thr, 0.04, f" {'>' if direction == 'above' else '<'}{thr:g}",
            transform=ax.get_xaxis_transform(), ha="left", va="bottom",
            fontsize=8, color=GREY)


def _dot_legend(fig, entries, y=0.945):
    handles = [Line2D([], [], marker="o", ls="none", ms=9, color=c, label=l)
               for l, c in entries]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y),
               ncol=len(entries), frameon=False, fontsize=11, handletextpad=0.4,
               columnspacing=2.2)


def _footnote(fig, text, y=0.012):
    fig.text(0.5, y, text, ha="center", va="bottom", fontsize=8.5, color="#666666")


# ---------------------------------------------------------------------------------------
# 1. metric distributions, stage_1 vs stage_2
# ---------------------------------------------------------------------------------------

def fig_metrics(results, out_path, run_name):
    fig, axes = plt.subplots(3, 4, figsize=(19, 11.5))
    for ax, (col, label, thr, direction) in zip(axes.ravel(), METRICS):
        for stage in ("stage_1", "stage_2"):
            _kde(ax, results.loc[results.stage == stage, col],
                 STAGE_COLOR[stage], STAGE_LABEL[stage])
        _threshold(ax, thr, direction)
        ax.set_title(label)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    for ax in axes.ravel()[len(METRICS):]:
        ax.set_visible(False)

    fig.suptitle(f"mosaic_scan metric distributions — stage 1 vs stage 2   ({run_name})",
                 fontsize=15, y=0.985)
    _dot_legend(fig, [(STAGE_LABEL[s], STAGE_COLOR[s]) for s in ("stage_1", "stage_2")])
    n_t = results.target.nunique()
    n_a = results.groupby("target").attempt.nunique().max()
    _footnote(fig, f"n = {len(results)} rows ({n_t} targets × {n_a} attempts × 2 stages). "
                   f"KDE, Silverman bandwidth. Dashed lines are the pipeline's pass thresholds.")
    fig.tight_layout(rect=[0, 0.025, 1, 0.925])
    path = os.path.join(out_path, "mosaic_scan_metrics.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------------------
# 2. per-target benchmark
# ---------------------------------------------------------------------------------------

def fig_targets(results, success, out_path, run_name):
    attempts = results.groupby("target").attempt.nunique()
    if success is not None and len(success):
        yield_n = success.groupby("target").size().reindex(attempts.index, fill_value=0)
    else:  # fall back to per-row passes if success.csv is absent
        yield_n = results[results.passed].groupby(["target", "attempt"]).size().gt(0) \
                    .groupby("target").sum().reindex(attempts.index, fill_value=0)
    rate = (yield_n / attempts).sort_values()
    order = list(rate.index)

    panel_metrics = [
        ("ipsae",             "ipSAE",                        0.75, "above"),
        ("plddt",             "Binder pLDDT",                 0.80, "above"),
        ("target_binder_pae", "target→binder pAE (Å)",  5.0, "below"),
        ("rmsd_to_blueprint", "CA-RMSD to blueprint (Å)", None, None),
        ("dssp_identity",     "DSSP identity vs blueprint",   None, None),
    ]

    fig = plt.figure(figsize=(20, 9.6))
    gs = fig.add_gridspec(1, 1 + len(panel_metrics), width_ratios=[1.45] + [1] * len(panel_metrics),
                          wspace=0.28, left=0.115, right=0.99, top=0.845, bottom=0.085)

    # --- yield bars ---
    ax = fig.add_subplot(gs[0, 0])
    ypos = np.arange(len(order))
    ax.barh(ypos, rate.values * 100, color=BLUE, alpha=0.8, height=0.68)
    for y, (r, n) in enumerate(zip(rate.values, yield_n.reindex(order).values)):
        ax.text(r * 100 + 1.2, y, f"{n}/{attempts[order[y]]}", va="center", fontsize=8.5,
                color="#444444")
    ax.set_yticks(ypos)
    ax.set_yticklabels(
        [f"{t if len(t) <= 20 else t[:18] + '…'}  "
         f"({results.loc[results.target == t, 'target_length'].iloc[0]} aa)" for t in order],
        fontsize=9)
    ax.set_xlabel("designs passing all filters (%)")
    ax.set_xlim(0, 105)
    ax.set_title("Design yield per target")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # --- per-target metric spreads ---
    for j, (col, label, thr, direction) in enumerate(panel_metrics):
        ax = fig.add_subplot(gs[0, j + 1])
        data = [results.loc[results.target == t, col].to_numpy() for t in order]
        bp = ax.boxplot(data, orientation="horizontal", positions=ypos, widths=0.62,
                        showfliers=False, patch_artist=True,
                        medianprops=dict(color="black", lw=1.3),
                        whiskerprops=dict(color="#999999"),
                        capprops=dict(color="#999999"),
                        boxprops=dict(edgecolor="#999999", lw=0.8))
        # neutral boxes: green is reserved for "this design passed", below
        for patch in bp["boxes"]:
            patch.set_facecolor("#E4E4E4")
        rng = np.random.default_rng(0)
        for y, t in enumerate(order):
            sub = results[results.target == t]
            ax.scatter(sub[col], y + rng.uniform(-0.19, 0.19, len(sub)),
                       s=16, alpha=0.75, linewidths=0, zorder=3,
                       color=np.where(sub.passed.to_numpy(), GREEN, "#9A9A9A"))
        if thr is not None:
            ax.axvline(thr, color=GREY, ls="--", lw=1.1, zorder=0)
        ax.set_yticks(ypos)
        ax.set_yticklabels([])
        ax.set_title(label)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_ylim(-0.7, len(order) - 0.3)

    fig.suptitle(f"mosaic_scan per-target benchmark   ({run_name})", fontsize=15, y=0.972)
    _dot_legend(fig, [("design passes all filters", GREEN), ("design fails ≥1 filter", "#B0B0B0")],
                y=0.925)
    _footnote(fig, "Targets ordered by yield. Each point is one design (both stages shown); "
                   "boxes are quartiles with the median. Dashed lines are pass thresholds.",
              y=0.022)
    path = os.path.join(out_path, "mosaic_scan_targets.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path, rate, yield_n, attempts


# ---------------------------------------------------------------------------------------
# 3. stage_1 vs stage_2, paired by (target, attempt)
# ---------------------------------------------------------------------------------------

def fig_stage_scatter(results, out_path, run_name):
    wide = results.pivot_table(index=["target", "attempt"], columns="stage",
                               values=[m[0] for m in METRICS])
    pairs = [(c, l, thr, d) for c, l, thr, d in METRICS
             if (c, "stage_1") in wide.columns and (c, "stage_2") in wide.columns]

    fig, axes = plt.subplots(3, 4, figsize=(19, 13.5))
    for ax, (col, label, thr, direction) in zip(axes.ravel(), pairs):
        x = wide[(col, "stage_1")].to_numpy(float)
        y = wide[(col, "stage_2")].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        ax.scatter(x, y, s=18, alpha=0.45, color=BLUE, linewidths=0)
        lo = min(x.min(), y.min()); hi = max(x.max(), y.max())
        pad = 0.05 * (hi - lo + 1e-9)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls="--", lw=1.0, color="#BBBBBB", zorder=0)
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
        r = pearsonr(x, y)[0] if x.size > 2 and x.std() > 0 and y.std() > 0 else np.nan
        above, below = int((y > x).sum()), int((y < x).sum())
        n = max(1, x.size)
        # which side of the diagonal is "better" depends on the metric's direction
        better_up = {"above": "stage 2 better", "below": "stage 1 better", None: ""}[direction] \
            if direction is not None else ""
        better_dn = {"above": "stage 1 better", "below": "stage 2 better", None: ""}[direction] \
            if direction is not None else ""
        ax.text(0.03, 0.97, f"above: {above} ({100*above/n:.0f}%)" + (f"\n{better_up}" if better_up else ""),
                transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox=dict(fc="#E8EEF7", ec="none", pad=2.5))
        ax.text(0.97, 0.03, f"below: {below} ({100*below/n:.0f}%)" + (f"\n{better_dn}" if better_dn else ""),
                transform=ax.transAxes, va="bottom", ha="right", fontsize=8,
                bbox=dict(fc="#FBEBE0", ec="none", pad=2.5))
        # ties are common for the discrete blueprint metrics -- say so rather than
        # letting above+below silently fail to reach 100%
        ties = n - above - below
        if ties:
            ax.text(0.97, 0.97, f"tied: {ties} ({100*ties/n:.0f}%)",
                    transform=ax.transAxes, va="top", ha="right", fontsize=8, color="#777777")
        ax.set_title(f"{label}\nr = {r:.2f}" if np.isfinite(r) else label)
        ax.set_xlabel("stage 1"); ax.set_ylabel("stage 2")
    for ax in axes.ravel()[len(pairs):]:
        ax.set_visible(False)

    fig.suptitle("Stage 1 vs stage 2: same salad blueprint, same attempt, "
                 "before vs after logit hardening\n"
                 f"(n = {len(wide)} paired designs; point above the diagonal = stage 2 scored higher)   "
                 f"({run_name})", fontsize=14, y=0.985)
    fig.tight_layout(rect=[0, 0.015, 1, 0.93])
    path = os.path.join(out_path, "mosaic_scan_stage_scatter.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------------------
# 4. cross-metric relationships
# ---------------------------------------------------------------------------------------

CROSS = [
    ("recovery",          "Self-consistency recovery", "rmsd_to_blueprint", "CA-RMSD to blueprint (Å)"),
    ("recovery",          "Self-consistency recovery", "dssp_identity",     "DSSP identity vs blueprint"),
    ("rmsd_to_blueprint", "CA-RMSD to blueprint (Å)", "dssp_identity",   "DSSP identity vs blueprint"),
    ("plddt",             "Binder pLDDT",              "within_binder_pae", "within-binder pAE (Å)"),
    ("plddt",             "Binder pLDDT",              "ipsae",             "ipSAE"),
    ("ipsae",             "ipSAE",                     "target_binder_pae", "target→binder pAE (Å)"),
    ("iptm",              "ipTM",                      "ipsae",             "ipSAE"),
    ("binder_target_contacts", "Binder–target contacts (entropy)", "ipsae", "ipSAE"),
    ("rmsd_to_blueprint", "CA-RMSD to blueprint (Å)", "ipsae",         "ipSAE"),
    ("target_length",     "Target length (aa)",        "ipsae",             "ipSAE"),
]


def fig_crossmetric(results, out_path, run_name):
    fig, axes = plt.subplots(2, 5, figsize=(21, 9.5))
    for ax, (xc, xl, yc, yl) in zip(axes.ravel(), CROSS):
        for mask, color, lab in [(~results.passed, "#C3C3C3", "fail"),
                                 (results.passed, GREEN, "pass")]:
            sub = results[mask]
            ax.scatter(sub[xc], sub[yc], s=20, alpha=0.5 if lab == "fail" else 0.75,
                       color=color, linewidths=0, label=lab)
        x = results[xc].to_numpy(float); y = results[yc].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        r = pearsonr(x[ok], y[ok])[0] if ok.sum() > 2 else np.nan
        # 3 decimals near |r|=1 so a near-perfect correlation doesn't read as exactly 1.00
        ax.set_title(f"r = {r:.3f}" if abs(r) > 0.99 else f"r = {r:.2f}", fontsize=10)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
    for ax in axes.ravel()[len(CROSS):]:
        ax.set_visible(False)

    fig.suptitle(f"Cross-metric relationships, all {len(results)} designs   ({run_name})",
                 fontsize=15, y=0.985)
    _dot_legend(fig, [("passes all filters", GREEN), ("fails ≥1 filter", "#C3C3C3")], y=0.935)
    _footnote(fig, "Pearson r computed over all designs (both stages), pass and fail together.")
    fig.tight_layout(rect=[0, 0.025, 1, 0.915])
    path = os.path.join(out_path, "mosaic_scan_crossmetric.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_path", default="runs/mosaic_scan_L80/")
    ap.add_argument("--out_path", default=None, help="defaults to --run_path")
    args = ap.parse_args()
    out_path = args.out_path or args.run_path
    os.makedirs(out_path, exist_ok=True)
    run_name = os.path.basename(os.path.normpath(args.run_path))

    results, success = load(args.run_path)
    print(f"loaded {len(results)} rows, {results.target.nunique()} targets, "
          f"{len(success) if success is not None else 0} successes")

    p1 = fig_metrics(results, out_path, run_name)
    p2, rate, yield_n, attempts = fig_targets(results, success, out_path, run_name)
    p3 = fig_stage_scatter(results, out_path, run_name)
    p4 = fig_crossmetric(results, out_path, run_name)
    for p in (p1, p2, p3, p4):
        print("wrote", p)

    print("\nper-target yield:")
    for t in rate.sort_values(ascending=False).index:
        print(f"  {t:30s} {yield_n[t]:2d}/{attempts[t]:2d}  ({100*rate[t]:5.1f}%)")
    print(f"  {'TOTAL':30s} {int(yield_n.sum()):2d}/{int(attempts.sum()):2d} "
          f"({100*yield_n.sum()/attempts.sum():5.1f}%)")


if __name__ == "__main__":
    main()
