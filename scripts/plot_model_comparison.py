"""Compare structure-prediction models on fold steering: does a ProteinMPNN sequence read off
a salad blueprint reliably re-fold into that blueprint under each model's hallucination?

Focuses on the two blueprint-fidelity columns every mosaic_scan run writes:
  rmsd_to_blueprint   CA-RMSD, binder-only superposition -- shape agreement
  dssp_identity       per-residue secondary-structure agreement -- shape agreement

Produces one figure, mosaic_model_comparison.png, written to --out_path:
  top row    KDEs of dssp_identity and rmsd_to_blueprint, one line per model, plus their
             joint scatter (does high fold recovery track low RMSD, and does that relationship
             hold the same way for every model?)
  bottom row per-target median of each metric, one row per target, so a model that wins overall
             but loses on specific targets is visible rather than averaged away

Usage:
    python3 scripts/plot_model_comparison.py \\
        --runs runs/mosaic/mosaic_scan_n30:joltz \\
               runs/mosaic_n20_protenixV1:protenix_v1 \\
               runs/mosaic_n20_protenixV2:protenix_v2 \\
        --out_path runs/model_comparison/

Each --runs entry is `path[:label]`; the label defaults to the directory's basename.
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

# Same house palette as plot_mosaic_scan.py (seaborn "deep"-like), extended with a third hue
# for the third model. GREY stays reserved for structural/neutral marks (thresholds, ties).
BLUE, ORANGE, GREEN, GREY = "#4C72B0", "#DD8452", "#55A868", "#8C8C8C"
MODEL_PALETTE = [BLUE, ORANGE, GREEN, "#C44E52", "#8172B3", "#937860"]

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

DEFAULT_RUNS = [
    "runs/mosaic/mosaic_scan_n30:joltz",
    "runs/mosaic/mosaic_n20_protenixV1:protenix_v1",
    "runs/mosaic_n20_protenixV2:protenix_v2",
]


def _kde(ax, values, color, label, fill=True):
    """Silverman-bandwidth KDE, with a fallback for degenerate/tiny data (same as
    plot_mosaic_scan.py's _kde, duplicated to keep this script standalone)."""
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


def _dot_legend(fig, entries, y=0.93):
    handles = [Line2D([], [], marker="o", ls="none", ms=9, color=c, label=l)
               for l, c in entries]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y),
               ncol=len(entries), frameon=False, fontsize=11, handletextpad=0.4,
               columnspacing=2.2)


def _footnote(fig, text, y=0.028):
    fig.text(0.5, y, text, ha="center", va="bottom", fontsize=8.5, color="#666666")


def _parse_runs(specs):
    out = []
    seen = set()
    for spec in specs:
        path, _, label = spec.partition(":")
        label = label or os.path.basename(os.path.normpath(path))
        assert label not in seen, f"duplicate model label '{label}' -- give --runs distinct labels"
        seen.add(label)
        out.append((path, label))
    return out


def load(runs):
    """One combined DataFrame with a `model` column, taken from results.csv in each run dir.
    Missing rmsd_to_blueprint/dssp_identity (e.g. a run with no --sample_structure blueprint)
    make that model silently absent from the corresponding panel, not an error."""
    frames = []
    for path, label in runs:
        p = os.path.join(path, "results.csv")
        d = pd.read_csv(p)
        d["passed"] = d["passed"].astype(str).str.strip() == "True"
        d["model"] = label
        frames.append(d)
        n_t = d.target.nunique()
        n_a = int(d.groupby("target").attempt.nunique().max())
        print(f"{label:16s} {p}: {len(d)} rows, {n_t} targets, up to {n_a} attempts/target")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    missing_cols = [c for c in ("rmsd_to_blueprint", "dssp_identity") if c not in combined]
    assert not missing_cols, (
        f"results.csv is missing {missing_cols} -- these runs weren't generated with "
        f"--sample_structure True (no salad blueprint to compare against)")
    return combined


def fig_comparison(df, models, colors, out_path):
    fig = plt.figure(figsize=(19, 15))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 2.1], hspace=0.38, wspace=0.28,
                          left=0.14, right=0.985, top=0.87, bottom=0.075)

    # ---- top-left / top-middle: KDEs, one line per model -------------------------------
    for col, label, ax_pos in [
            ("dssp_identity", "DSSP identity vs blueprint", gs[0, 0]),
            ("rmsd_to_blueprint", "CA-RMSD to blueprint (Å)", gs[0, 1])]:
        ax = fig.add_subplot(ax_pos)
        for model, color in zip(models, colors):
            _kde(ax, df.loc[df.model == model, col], color, model)
        ax.set_title(label)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)

    # ---- top-right: joint scatter, colored by model, with per-model r ------------------
    ax = fig.add_subplot(gs[0, 2])
    lines = []
    for model, color in zip(models, colors):
        sub = df[df.model == model]
        ax.scatter(sub.dssp_identity, sub.rmsd_to_blueprint, s=14, alpha=0.45,
                  color=color, linewidths=0)
        x, y = sub.dssp_identity.to_numpy(float), sub.rmsd_to_blueprint.to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        r = pearsonr(x[ok], y[ok])[0] if ok.sum() > 2 and x[ok].std() > 0 else np.nan
        lines.append(f"{model}: r = {r:.2f}" if np.isfinite(r) else model)
    ax.set_xlabel("DSSP identity vs blueprint")
    ax.set_ylabel("CA-RMSD to blueprint (Å)")
    ax.set_title("Fold recovery vs fold deviation")
    ax.text(0.98, 0.97, "\n".join(lines), transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, bbox=dict(fc="white", ec=GREY, lw=0.6, pad=4))

    # ---- bottom row: per-target medians, one wide panel per metric ---------------------
    order = (df.groupby("target").dssp_identity.median()
            .sort_values(ascending=False).index.tolist())
    lengths = df.groupby("target").target_length.first()
    ypos = {t: i for i, t in enumerate(order)}

    for col, label, ax_pos, lo_hi in [
            ("dssp_identity", "Median DSSP identity vs blueprint", gs[1, 0], (0, 1.02)),
            ("rmsd_to_blueprint", "Median CA-RMSD to blueprint (Å)", gs[1, 1], None)]:
        ax = fig.add_subplot(ax_pos)
        for model, color in zip(models, colors):
            g = df[df.model == model].groupby("target")[col].median()
            g = g.reindex(order)
            y = [ypos[t] for t in order if pd.notna(g.get(t))]
            x = [g[t] for t in order if pd.notna(g.get(t))]
            ax.scatter(x, y, s=48, color=color, label=model, zorder=3,
                      edgecolors="white", linewidths=0.6)
        # connect the per-target dots across models so a target's spread reads at a glance
        for t in order:
            vals = [df[(df.model == m) & (df.target == t)][col].median() for m in models]
            vals = [v for v in vals if pd.notna(v)]
            if len(vals) > 1:
                ax.plot([min(vals), max(vals)], [ypos[t], ypos[t]], color=GREY, lw=0.8,
                       alpha=0.6, zorder=1)
        ax.set_yticks(range(len(order)))
        if col == "dssp_identity":
            # Labels once, on the leftmost panel only -- both share the same row order, so a
            # second copy on the RMSD panel would just repeat them at the cost of plot width.
            ax.set_yticklabels(
                [f"{t if len(t) <= 22 else t[:20] + '…'}  ({int(lengths[t])} aa)"
                 for t in order], fontsize=8.5)
        else:
            ax.set_yticklabels([])
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_xlabel(label)
        ax.set_title(label)
        if lo_hi:
            ax.set_xlim(*lo_hi)

    # third bottom slot: which model wins each target on dssp_identity, as a quick summary
    ax = fig.add_subplot(gs[1, 2])
    wins = {}
    for t in order:
        meds = {m: df[(df.model == m) & (df.target == t)].dssp_identity.median() for m in models}
        meds = {m: v for m, v in meds.items() if pd.notna(v)}
        if meds:
            wins[t] = max(meds, key=meds.get)
    win_counts = pd.Series(wins).value_counts().reindex(models, fill_value=0)
    ax.barh(range(len(models)), win_counts.values, color=colors, height=0.6)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel("targets with the highest median\nDSSP identity (of "
                  f"{len(order)} targets)")
    ax.set_title("Fold-recovery winner, per target")
    ax.invert_yaxis()
    for i, v in enumerate(win_counts.values):
        ax.text(v + 0.05, i, str(int(v)), va="center", fontsize=9)

    fig.suptitle("Fold steering across structure-prediction models: does the ProteinMPNN "
                 "sequence keep the salad blueprint's fold?", fontsize=14.5, y=0.975)
    _dot_legend(fig, list(zip(models, colors)), y=0.905)
    _footnote(fig,
             "rmsd_to_blueprint / dssp_identity are BINDER-ONLY comparisons -- they measure "
             "whether the hallucinated fold matches the salad blueprint's fold, not where the "
             "binder sits on the target.\n"
             f"n = {len(df)} rows across {len(models)} models, both stages pooled. Per-target "
             "panels use each model's median over its own attempts.")
    path = os.path.join(out_path, "mosaic_model_comparison.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS,
                    help="run_path[:label] entries, one per model")
    ap.add_argument("--out_path", default="runs/model_comparison")
    args = ap.parse_args()
    os.makedirs(args.out_path, exist_ok=True)

    runs = _parse_runs(args.runs)
    df = load(runs)
    models = [label for _, label in runs]
    colors = MODEL_PALETTE[:len(models)]

    print()
    print(df.groupby("model")[["dssp_identity", "rmsd_to_blueprint"]]
          .median().round(3).to_string())
    print()

    path = fig_comparison(df, models, colors, args.out_path)
    print("wrote", path)


if __name__ == "__main__":
    main()
