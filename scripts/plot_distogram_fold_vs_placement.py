"""Analyse the mosaic_scan blueprint-distogram-loss weight sweeps: does the loss trade fold
recovery for placement recovery of the salad blueprint, and does that trade differ per structure
model?

Reads results.csv from one --distogram_loss_weights sweep run per model (default: the three
known sweeps -- joltz/Boltz-2, Protenix v1, Protenix v2) and, per model, plots ONE scatter with
every target pooled together:

    x = placement recovery  (epitope_recall_vs_blueprint: fraction of the blueprint's docking
                              epitope that the hallucinated complex actually contacts)
    y = fold recovery       (dssp_identity: secondary-structure identity of the predicted binder
                              to the salad blueprint)

one point per (target, attempt), colored by --distogram_loss_weight. Also writes/prints a
per-model, per-weight summary table (mean/median of both axes, n).

Usage:
    python3 scripts/plot_distogram_fold_vs_placement.py
    python3 scripts/plot_distogram_fold_vs_placement.py --stage stage_1
    python3 scripts/plot_distogram_fold_vs_placement.py \\
        --runs joltz=runs/mosaic_distogram_sweep protenix_v1=runs/mosaic_distogram_sweep_protenixV1
"""

import os
import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_RUNS = {
    # runs/ has been reorganized by hand since this script was first written -- the plain joltz
    # (weights 0,1,2) sweep now lives under runs/mosaic/; the Protenix ones are still top-level.
    # Update these paths again if they move; a missing one is skipped with a warning, not an error.
    "joltz": "runs/mosaic/mosaic_distogram_sweep_joltz",
    "protenix_v1": "runs/mosaic_distogram_sweep_protenixV1",
    "protenix_v2": "runs/mosaic_distogram_sweep_protenixV2",
}
MODEL_TITLE = {
    "joltz": "Boltz-2 (joltz)", "protenix_v1": "Protenix v1", "protenix_v2": "Protenix v2",
    "boltz_ext": "Boltz-2 (joltz) — extended weights 3–5",
}

X_COL, X_LABEL = "epitope_recall_vs_blueprint", "Placement recovery\n(fraction of blueprint epitope contacted)"
Y_COL, Y_LABEL = "dssp_identity", "Fold recovery\n(DSSP identity to blueprint)"

INK = "#2B2B2B"
# distogram_loss_weight is an ORDERED magnitude, not an unordered category, so it gets a single-hue
# ordinal ramp (light -> dark = more weight), not arbitrary categorical hues. Validated with
# dataviz's validate_palette.js --ordinal (monotone lightness, >=0.06 adjacent steps, light end
# clears 2:1 contrast on white). Marker shape is a second, color-independent encoding of the same
# variable, since the lightest ramp step is a borderline 2.06:1 against a white figure background.
RAMP = ["#86b6ef", "#2a78d6", "#104281"]
MARKERS = ["o", "s", "^"]

KEYS = ["target", "attempt", "stage", "distogram_weight"]


def load_run(path, label):
    fp = os.path.join(path, "results.csv")
    if not os.path.isfile(fp):
        raise SystemExit(f"no results.csv in {path}")
    df = pd.read_csv(fp)
    # A --resume run duplicates the row for an (attempt, weight) that was interrupted mid-way and
    # redone; keep the row from the run that actually finished it (see compare_distogram_loss.py's
    # load_arm, same fix).
    df = df.drop_duplicates(subset=KEYS, keep="last")
    if "structure_model" in df and not (df["structure_model"] == label).all():
        seen = sorted(df["structure_model"].unique())
        print(f"WARNING [{label}]: {fp} was written with structure_model {seen}, not '{label}' -- "
              f"check --runs against the right directory.")
    for col in (X_COL, Y_COL, "distogram_weight"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def weight_styles(weights):
    if len(weights) > len(RAMP):
        raise SystemExit(f"{len(weights)} distinct weights {weights} but only {len(RAMP)} ramp "
                         f"steps are validated -- extend RAMP/MARKERS and re-validate with "
                         f"dataviz's validate_palette.js --ordinal before adding more.")
    return {w: (RAMP[i], MARKERS[i]) for i, w in enumerate(weights)}


def plot_model(label, df, stage, out_path, rng):
    sub = df[df.stage == stage].dropna(subset=[X_COL, Y_COL, "distogram_weight"])
    if sub.empty:
        print(f"[{label}] no {stage} rows with both {X_COL} and {Y_COL}; skipped")
        return None, None
    weights = sorted(sub.distogram_weight.unique())
    styles = weight_styles(weights)

    fig, ax = plt.subplots(figsize=(7.4, 6.8))
    for w in weights:
        wsub = sub[sub.distogram_weight == w]
        color, marker = styles[w]
        # Many (target, attempt) pairs land on the exact same fraction (both axes are k/n over a
        # small blueprint epitope / DSSP length), so points would stack exactly on top of each
        # other; a small fixed-seed jitter (display only) spreads them for legibility.
        # Clip AFTER jitter: both axes are bounded fractions ([0, 1]), and letting the jitter push
        # a point past that bound would visually claim e.g. "DSSP identity > 1", which is nonsense.
        jx = np.clip(wsub[X_COL] + rng.uniform(-0.008, 0.008, len(wsub)), 0.0, 1.0)
        jy = np.clip(wsub[Y_COL] + rng.uniform(-0.008, 0.008, len(wsub)), 0.0, 1.0)
        ax.scatter(jx, jy, s=42, color=color, marker=marker,
                   edgecolor="white", linewidth=0.7, alpha=0.85, zorder=3,
                   label=f"weight {w:g}  (n={len(wsub)})")
        ax.scatter(wsub[X_COL].mean(), wsub[Y_COL].mean(), s=240, color=color, marker=marker,
                   edgecolor=INK, linewidth=1.8, zorder=4, label="_nolegend_")
    pad_y = 0.05 * max(sub[Y_COL].max() - sub[Y_COL].min(), 0.05)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(max(0.0, sub[Y_COL].min() - pad_y), min(1.03, sub[Y_COL].max() + pad_y))
    ax.set_xlabel(X_LABEL, fontsize=10.5)
    ax.set_ylabel(Y_LABEL, fontsize=10.5)
    ax.set_title(MODEL_TITLE.get(label, label), fontsize=13.5, fontweight="bold")
    ax.grid(color="#E6E6E6", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5,
             title="distogram loss weight (small = design, large outline = mean)",
             title_fontsize=8.5)
    n_t, n_pts = sub.target.nunique(), len(sub)
    fig.text(0.5, 0.006,
             f"{label}: {n_pts} (target, attempt) points across {n_t} targets, {stage}. "
             f"Points jittered slightly to reduce exact overlap.",
             ha="center", va="bottom", fontsize=8.2, color="#666666")
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    path = os.path.join(out_path, f"fold_vs_placement_{label}.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path, sub


def summarize(label, sub, stage):
    rows = []
    for w, wsub in sub.groupby("distogram_weight"):
        rows.append(dict(
            model=label, stage=stage, distogram_weight=w, n=len(wsub), n_targets=wsub.target.nunique(),
            mean_fold_recovery=wsub[Y_COL].mean(), median_fold_recovery=wsub[Y_COL].median(),
            mean_placement_recovery=wsub[X_COL].mean(), median_placement_recovery=wsub[X_COL].median()))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=None,
                    help="label=path pairs, e.g. joltz=runs/x protenix_v1=runs/y "
                         "(default: the three known --distogram_loss_weights sweeps)")
    ap.add_argument("--stage", default="stage_2", help="stage to plot (default: stage_2, the "
                    "hardened final design; pass stage_1 for the soft-APGM intermediate)")
    ap.add_argument("--seed", type=int, default=0, help="display-jitter seed")
    ap.add_argument("--out_path", default="runs/mosaic/mosaic_distogram_analysis",
                    help="where to write the PNGs and summary CSV")
    args = ap.parse_args()
    runs = DEFAULT_RUNS if not args.runs else dict(r.split("=", 1) for r in args.runs)
    os.makedirs(args.out_path, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    summaries = []
    for label, path in runs.items():
        if not os.path.isdir(path):
            print(f"[{label}] {path} does not exist; skipped")
            continue
        df = load_run(path, label)
        fig_path, sub = plot_model(label, df, args.stage, args.out_path, rng)
        if sub is None:
            continue
        print(f"[{label}] figure: {fig_path}")
        summary = summarize(label, sub, args.stage)
        summaries.append(summary)
        with pd.option_context("display.float_format", "{:.3f}".format):
            print(summary.drop(columns=["model", "stage"]).to_string(index=False))
        print()

    if summaries:
        out = pd.concat(summaries, ignore_index=True)
        # Only the default (no --runs) call reuses the plain filename -- a custom --runs subset
        # gets its own file, named after the labels it actually plotted, so e.g. a one-off extra
        # run never clobbers the combined summary from a previous default run.
        name = ("fold_vs_placement_summary.csv" if not args.runs else
                f"fold_vs_placement_summary_{'_'.join(sorted(runs))}.csv")
        csv_path = os.path.join(args.out_path, name)
        out.to_csv(csv_path, index=False)
        print(f"summary: {csv_path}")


if __name__ == "__main__":
    main()
