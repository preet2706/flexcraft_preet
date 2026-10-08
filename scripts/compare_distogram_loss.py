"""Compare two mosaic_scan runs that differ only in --distogram_loss (off vs on).

Built for bash_scripts/mosaic_distogram_test.sh, where both arms dock the SAME blueprints (attempt i
is seeded from (seed, target, attempt) alone), so every attempt is a matched pair. Produces, in
--out_path (default: the parent directory of --on):

  distogram_ab_summary.csv   one row per (stage, metric): means, paired delta, wins, Wilcoxon p
  distogram_ab.png           paired slope plot of the headline metrics (one line per blueprint)

and prints the same table. Everything is CPU-only.

Usage:
    python3 scripts/compare_distogram_loss.py \
        --off runs/mosaic_distogram_test/off --on runs/mosaic_distogram_test/on

Weight sweep (one mosaic_scan run made with --distogram_loss_weights, all weights on the same
blueprints): every weight is paired to the baseline weight, per (attempt, stage):
    python3 scripts/compare_distogram_loss.py --sweep runs/mosaic_distogram_sweep [--baseline 0.0]
which writes distogram_sweep_summary.csv and distogram_sweep.png into the run directory.

With ~10 pairs the Wilcoxon test has little power: read the paired deltas and win counts first, and
treat p-values as a sanity check, not as the result.
"""

import os
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# House palette from plot_mosaic_scan.py (validated: CVD dE 20, normal dE 27). The orange is below 3:1
# contrast on white, so every panel is directly labelled and arms also differ by marker shape.
BLUE, ORANGE, GREY, INK = "#4C72B0", "#DD8452", "#8C8C8C", "#2B2B2B"
ARM_COLOR = {"off": BLUE, "on": ORANGE}
ARM_MARKER = {"off": "o", "on": "s"}
ARM_LABEL = {"off": "no distogram loss", "on": "distogram loss"}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 8.5,
    "font.size": 10,
})

# (column, label, better) -- `better` is the direction that counts as an improvement.
PLACEMENT = [
    ("epitope_jaccard",             "Epitope Jaccard vs blueprint",      "higher"),
    ("epitope_jaccard_10a",         "Epitope Jaccard (10 Å cutoff)",     "higher"),
    ("epitope_recall_vs_blueprint", "Blueprint epitope recovered",       "higher"),
    ("hotspot_recall",              "Requested hotspots contacted",      "higher"),
    ("binder_rmsd_target_frame",    "Binder RMSD, target frame (Å)",     "lower"),
    ("binder_centroid_drift",       "Binder centroid drift (Å)",         "lower"),
    # only present in runs made with --coldspot_file; lower = better (0 = fully avoided)
    ("coldspot_hit_predicted",      "Coldspot residues contacted",       "lower"),
]
QUALITY = [
    ("ipsae",                "ipSAE",                          "higher"),
    ("iptm",                 "ipTM",                           "higher"),
    ("plddt",                "Binder pLDDT",                   "higher"),
    ("binder_target_pae",    "binder→target pAE (Å)",     "lower"),
    ("rmsd_to_blueprint",    "CA-RMSD to blueprint (Å)",  "lower"),
]
# Present in every row of a sweep (incl. the baseline weight); a manipulation check.
DIAGNOSTIC = [("distogram_nll", "Distogram NLL vs blueprint", "lower")]
METRICS = PLACEMENT + QUALITY + DIAGNOSTIC
METRIC_INFO = {col: (label, better) for col, label, better in METRICS}
# Figure panels: placement on the top row, design quality below.
FIG_METRICS = [PLACEMENT[0], PLACEMENT[2], PLACEMENT[4], QUALITY[0], QUALITY[2], QUALITY[4]]
KEYS = ["target", "attempt", "stage"]


def load_arm(run_path):
    path = os.path.join(run_path, "results.csv")
    if not os.path.isfile(path):
        raise SystemExit(f"no results.csv in {run_path}")
    df = pd.read_csv(path)
    # An attempt interrupted after stage 1 and re-run under --resume writes its stage_1 row twice;
    # keep the last one (from the run that completed) so the pairing below is one row per key.
    df = df.drop_duplicates(
        subset=KEYS + (["distogram_weight"] if "distogram_weight" in df else []), keep="last")
    df["passed"] = df["passed"].astype(str).str.strip() == "True"
    for col, _, _ in METRICS:
        if col in df:  # columns hold the string "none" when a metric was not computed
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def pair(off, on, label=""):
    """Inner-join the arms on (target, attempt, stage); report what was dropped."""
    paired = off.merge(on, on=KEYS, suffixes=("_off", "_on"))
    n_off, n_on = len(off), len(on)
    if len(paired) != n_off or len(paired) != n_on:
        print(f"WARNING{label}: only {len(paired)} of {n_off} off-rows / {n_on} on-rows have a partner "
              f"(unfinished attempts are dropped from the comparison).")
    return paired


def _blueprint_ca(path):
    ca = []
    with open(path, "rt") as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                ca.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.asarray(ca)


def check_same_blueprints(paired, off_path, on_path):
    """The paired design is only valid if attempt i got the same blueprint in both arms.

    Compares the saved blueprint complexes (CA coordinates, so no alignment is needed as long as
    the run was deterministic) and, failing that, the cheap columns that depend only on the
    blueprint. Prints a warning, never aborts: salad on GPU need not be bit-reproducible.
    """
    bad = []
    for (target, attempt), _ in paired.groupby(["target", "attempt"]):
        f_off = os.path.join(off_path, "blueprints", f"{target}_{attempt}_complex.pdb")
        f_on = os.path.join(on_path, "blueprints", f"{target}_{attempt}_complex.pdb")
        if os.path.isfile(f_off) and os.path.isfile(f_on):
            a, b = _blueprint_ca(f_off), _blueprint_ca(f_on)
            dev = np.abs(a - b).max() if a.shape == b.shape else np.inf
            if dev > 0.05:
                bad.append((target, attempt, f"max CA deviation {dev:.2f} Å"))
    for col in ("n_epitope_blueprint", "blueprint_center_offset"):
        if f"{col}_off" in paired and f"{col}_on" in paired:
            diff = (paired[f"{col}_off"].astype(float) - paired[f"{col}_on"].astype(float)).abs()
            for i in np.flatnonzero(diff.to_numpy() > 1e-3):
                row = paired.iloc[i]
                bad.append((row["target"], row["attempt"], f"{col} differs"))
    if bad:
        print("WARNING: the arms did NOT get identical blueprints for these attempts, so they are "
              "not true pairs:")
        for target, attempt, why in sorted(set(bad)):
            print(f"  {target} attempt {attempt}: {why}")
    else:
        print("pairing check: blueprints identical in both arms for all "
              f"{paired.groupby(['target', 'attempt']).ngroups} attempts")


def summarize(paired):
    rows = []
    for stage, sub in paired.groupby("stage"):
        for col, label, better in METRICS:
            if f"{col}_off" not in sub:
                continue
            off, on = sub[f"{col}_off"].to_numpy(float), sub[f"{col}_on"].to_numpy(float)
            ok = np.isfinite(off) & np.isfinite(on)
            off, on = off[ok], on[ok]
            if off.size == 0:
                continue
            delta = on - off
            sign = 1.0 if better == "higher" else -1.0
            p = np.nan  # undefined when every delta is zero (identical values in both arms)
            if np.any(delta != 0):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    p = float(wilcoxon(delta).pvalue)
            rows.append(dict(
                stage=stage, metric=col, better=better, n=int(off.size),
                mean_off=off.mean(), mean_on=on.mean(), mean_delta=delta.mean(),
                improved=int((sign * delta > 0).sum()), worse=int((sign * delta < 0).sum()),
                wilcoxon_p=p))
        passed_off = int(sub["passed_off"].sum())
        passed_on = int(sub["passed_on"].sum())
        rows.append(dict(stage=stage, metric="passed (count)", better="higher", n=len(sub),
                         mean_off=passed_off, mean_on=passed_on, mean_delta=passed_on - passed_off,
                         improved=np.nan, worse=np.nan, wilcoxon_p=np.nan))
    return pd.DataFrame(rows)


def print_null(off_path, on_path):
    """Median shuffled-pair Jaccard: what epitope agreement chance alone gives."""
    for arm, path in (("off", off_path), ("on", on_path)):
        p = os.path.join(path, "placement_null.csv")
        if os.path.isfile(p):
            null = pd.read_csv(p)
            for stage, sub in null.groupby("stage"):
                j = pd.to_numeric(sub["epitope_jaccard"], errors="coerce")
                print(f"  null Jaccard [{arm}, {stage}]: median {j.median():.3f} "
                      f"(mean {j.mean():.3f}, n={len(j)} shuffled pairs)")


def print_nll(on_path):
    """distogram_nll only exists in the `on` arm, so it can only be followed within that arm."""
    p = os.path.join(on_path, "trajectory.csv")
    if not os.path.isfile(p):
        return
    traj = pd.read_csv(p)
    if "distogram_nll" not in traj:
        print("  (on-arm trajectory.csv has no distogram_nll column -- was --distogram_loss True set?)")
        return
    traj["distogram_nll"] = pd.to_numeric(traj["distogram_nll"], errors="coerce")
    first = traj[traj.step == traj.step.min()].distogram_nll.mean()
    last = traj[traj.step == traj.step.max()].distogram_nll.mean()
    print(f"  distogram NLL along the optimisation (on arm, mean over attempts): "
          f"step {traj.step.min()}: {first:.3f} -> step {traj.step.max()}: {last:.3f}")


def fig_paired(paired, stage, out_path):
    sub = paired[paired.stage == stage]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.6))
    for ax, (col, label, better) in zip(axes.ravel(), FIG_METRICS):
        if f"{col}_off" not in sub:
            ax.set_visible(False)
            continue
        off = sub[f"{col}_off"].to_numpy(float)
        on = sub[f"{col}_on"].to_numpy(float)
        ok = np.isfinite(off) & np.isfinite(on)
        off, on = off[ok], on[ok]
        for a, b in zip(off, on):
            ax.plot([0, 1], [a, b], color=GREY, lw=1.0, alpha=0.55, zorder=1)
        for x, vals, arm in ((0, off, "off"), (1, on, "on")):
            ax.scatter(np.full_like(vals, x), vals, s=46, color=ARM_COLOR[arm], marker=ARM_MARKER[arm],
                       edgecolor="white", linewidth=1.0, zorder=3)
        if off.size:
            ax.plot([0, 1], [off.mean(), on.mean()], color=INK, lw=2.2, zorder=2)
            ax.scatter([0, 1], [off.mean(), on.mean()], s=90, color="white", edgecolor=INK,
                       linewidth=2.0, zorder=4)
            improved = int(((on - off) * (1 if better == "higher" else -1) > 0).sum())
            ax.set_xlabel(f"mean Δ {on.mean() - off.mean():+.3g}   ·   {improved}/{off.size} improved",
                          fontsize=8.5, color="#555555")
        ax.set_xlim(-0.35, 1.35)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["off", "on"])
        ax.set_title(f"{label}  {'↑' if better == 'higher' else '↓'}")
        ax.grid(axis="y", color="#E6E6E6", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
    handles = [
        Line2D([], [], marker=ARM_MARKER[a], ls="none", ms=9, color=ARM_COLOR[a], label=ARM_LABEL[a])
        for a in ("off", "on")
    ] + [
        Line2D([], [], color=GREY, lw=1.2, label="one blueprint (same in both arms)"),
        Line2D([], [], color=INK, lw=2.2, marker="o", mfc="white", mec=INK, ms=8, label="mean"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=4,
               frameon=False, fontsize=10)
    n = sub.groupby(["target", "attempt"]).ngroups
    fig.suptitle(f"Blueprint-distogram loss, off vs on  —  {stage}", fontsize=14, y=0.995)
    fig.text(0.5, 0.008, f"{n} paired attempts. ↑/↓ = direction of improvement. Top row: does the binder "
                         f"stay where the blueprint was docked? Bottom row: design quality.",
             ha="center", va="bottom", fontsize=8.5, color="#666666")
    fig.tight_layout(rect=[0, 0.03, 1, 0.925])
    path = os.path.join(out_path, "distogram_ab.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def summarize_sweep(df, baseline):
    """Pair every non-baseline weight to the baseline weight (same target/attempt/stage) and reuse
    summarize(): in the result mean_off is the baseline, mean_on the weight, mean_delta on - off."""
    weights = sorted(df["distogram_weight"].dropna().unique())
    if baseline not in weights:
        raise SystemExit(f"baseline weight {baseline:g} is not in the run (weights: "
                         f"{[f'{w:g}' for w in weights]}); include 0.0 in --distogram_loss_weights "
                         f"or pass --baseline")
    base = df[df["distogram_weight"] == baseline].drop(columns="distogram_weight")
    parts = []
    for w in weights:
        if w == baseline:
            continue
        sub = df[df["distogram_weight"] == w].drop(columns="distogram_weight")
        paired = pair(base, sub, label=f" (weight {w:g})")
        if paired.empty:
            print(f"weight {w:g}: no attempt finished at both this weight and the baseline; skipped")
            continue
        part = summarize(paired)
        part.insert(1, "weight", w)
        parts.append(part)
    if not parts:
        raise SystemExit("no weight has any attempt paired with the baseline")
    return weights, pd.concat(parts, ignore_index=True)


def print_sweep_table(summary, baseline, stage):
    """metric x weight: `mean (delta vs baseline, #improved/n)`."""
    sub = summary[summary.stage == stage]
    order = [c for c, _, _ in METRICS] + ["passed (count)"]
    table = {}
    for _, r in sub.iterrows():
        row = table.setdefault(r.metric, {f"base {baseline:g}": f"{r.mean_off:.3f}"})
        wins = "" if np.isnan(r.improved) else f", {int(r.improved)}/{int(r.n)}"
        row[f"w={r.weight:g}"] = f"{r.mean_on:.3f} ({r.mean_delta:+.3f}{wins})"
    frame = pd.DataFrame(table).T
    frame = frame.reindex([m for m in order if m in frame.index])
    print(f"\n=== {stage}: mean (delta vs baseline weight {baseline:g}, blueprints improved/paired) ===")
    print(frame.fillna("").to_string())


def print_sweep_diagnostics(run_path):
    null_path = os.path.join(run_path, "placement_null.csv")
    if os.path.isfile(null_path):
        null = pd.read_csv(null_path)
        if "distogram_weight" in null:
            for (w, stage), sub in null.groupby(["distogram_weight", "stage"]):
                j = pd.to_numeric(sub["epitope_jaccard"], errors="coerce")
                print(f"  null Jaccard [w={w:g}, {stage}]: median {j.median():.3f} (n={len(j)} shuffled pairs)")
    traj_path = os.path.join(run_path, "trajectory.csv")
    if os.path.isfile(traj_path):
        traj = pd.read_csv(traj_path)
        if {"distogram_nll", "distogram_weight"} <= set(traj.columns):
            traj["distogram_nll"] = pd.to_numeric(traj["distogram_nll"], errors="coerce")
            for w, sub in traj.groupby("distogram_weight"):
                first = sub[sub.step == sub.step.min()].distogram_nll.mean()
                last = sub[sub.step == sub.step.max()].distogram_nll.mean()
                print(f"  distogram NLL over the optimisation [w={w:g}]: "
                      f"step {sub.step.min()}: {first:.3f} -> step {sub.step.max()}: {last:.3f}")


def _has(df, col):
    return col in df and pd.to_numeric(df[col], errors="coerce").notna().any()


def fig_sweep(df, stage, weights, baseline, out_path):
    sub = df[df.stage == stage]
    # Only blueprints that finished EVERY weight are drawn, so each mean averages the same
    # blueprints (the tables pair each weight to the baseline separately and use more rows).
    n_all = sub["attempt"].nunique()
    complete = sub.groupby("attempt")["distogram_weight"].nunique() == len(weights)
    sub = sub[sub["attempt"].isin(complete[complete].index)]
    if sub.empty:
        print("no blueprint has finished every weight yet; skipping the figure")
        return "(none)"
    spot = "coldspot_hit_predicted" if _has(sub, "coldspot_hit_predicted") else "hotspot_recall"
    cols = ["epitope_jaccard", spot, "binder_rmsd_target_frame", "distogram_nll",
            "ipsae", "iptm", "plddt", "rmsd_to_blueprint"]
    xs = np.arange(len(weights))
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.8))
    for ax, col in zip(axes.ravel(), cols):
        if not _has(sub, col):
            ax.set_visible(False)
            continue
        label, better = METRIC_INFO[col]
        wide = sub.pivot_table(index="attempt", columns="distogram_weight", values=col,
                               aggfunc="last").reindex(columns=weights)
        for _, row in wide.iterrows():
            ax.plot(xs, row.to_numpy(float), color=GREY, lw=1.0, alpha=0.55, zorder=1)
        for x, w in zip(xs, weights):
            vals = wide[w].dropna().to_numpy(float)
            arm = "off" if w == baseline else "on"
            ax.scatter(np.full_like(vals, x), vals, s=40, color=ARM_COLOR[arm], marker=ARM_MARKER[arm],
                       edgecolor="white", linewidth=1.0, zorder=3)
        means = wide.mean(axis=0).to_numpy(float)
        ax.plot(xs, means, color=INK, lw=2.2, zorder=2)
        ax.scatter(xs, means, s=80, color="white", edgecolor=INK, linewidth=2.0, zorder=4)
        ax.set_xlim(-0.4, len(weights) - 0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{w:g}" for w in weights])
        ax.set_xlabel("distogram loss weight", fontsize=8.5, color="#555555")
        ax.set_title(f"{label}  {'↑' if better == 'higher' else '↓'}")
        ax.grid(axis="y", color="#E6E6E6", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
    handles = [
        Line2D([], [], marker=ARM_MARKER["off"], ls="none", ms=9, color=ARM_COLOR["off"],
               label=f"baseline (weight {baseline:g})"),
        Line2D([], [], marker=ARM_MARKER["on"], ls="none", ms=9, color=ARM_COLOR["on"],
               label="distogram loss on"),
        Line2D([], [], color=GREY, lw=1.2, label="one blueprint (shared by every weight)"),
        Line2D([], [], color=INK, lw=2.2, marker="o", mfc="white", mec=INK, ms=8, label="mean"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=4,
               frameon=False, fontsize=10)
    n = sub["attempt"].nunique()
    partial = "" if n == n_all else f" (of {n_all}; the rest have not finished every weight)"
    fig.suptitle(f"Blueprint-distogram loss weight sweep  —  {stage}", fontsize=14, y=0.995)
    fig.text(0.5, 0.008, f"{n} blueprints{partial}, each hallucinated once per weight from the same starting "
                         f"sequence. ↑/↓ = direction of improvement. Top row: placement + the loss "
                         f"term itself; bottom row: design quality.",
             ha="center", va="bottom", fontsize=8.5, color="#666666")
    fig.tight_layout(rect=[0, 0.03, 1, 0.925])
    path = os.path.join(out_path, "distogram_sweep.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main_sweep(args):
    out_path = args.out_path or args.sweep
    os.makedirs(out_path, exist_ok=True)
    df = load_arm(args.sweep)
    if "distogram_weight" not in df:
        raise SystemExit(f"{args.sweep}/results.csv has no distogram_weight column: it was not made "
                         f"with --distogram_loss_weights")
    weights, summary = summarize_sweep(df, args.baseline)
    csv_path = os.path.join(out_path, "distogram_sweep_summary.csv")
    summary.to_csv(csv_path, index=False)
    for stage in sorted(summary.stage.unique(), reverse=True):  # stage_2 first
        print_sweep_table(summary, args.baseline, stage)
    print("\nreference / diagnostics:")
    print_sweep_diagnostics(args.sweep)
    if args.stage in set(df.stage):
        print(f"\nfigure : {fig_sweep(df, args.stage, weights, args.baseline, out_path)}")
    print(f"summary: {csv_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--off", help="run directory of the arm WITHOUT the distogram loss")
    ap.add_argument("--on", help="run directory of the arm WITH the distogram loss")
    ap.add_argument("--sweep", help="run directory of a --distogram_loss_weights sweep (instead of "
                                    "--off/--on)")
    ap.add_argument("--baseline", type=float, default=0.0,
                    help="sweep mode: the weight every other weight is compared to (default 0.0)")
    ap.add_argument("--out_path", default=None,
                    help="where to write outputs (default: parent of --on, or the sweep directory)")
    ap.add_argument("--stage", default="stage_2", help="stage shown in the figure (default: stage_2)")
    args = ap.parse_args()
    if args.sweep:
        if args.off or args.on:
            ap.error("--sweep cannot be combined with --off/--on")
        return main_sweep(args)
    if not (args.off and args.on):
        ap.error("give either --off and --on, or --sweep")
    out_path = args.out_path or os.path.dirname(os.path.abspath(args.on.rstrip("/")))
    os.makedirs(out_path, exist_ok=True)

    off, on = load_arm(args.off), load_arm(args.on)
    if "distogram_nll" in off.columns:
        print(f"WARNING: {args.off} has a distogram_nll column -- it looks like a run WITH the loss.")
    if "distogram_nll" not in on.columns:
        print(f"WARNING: {args.on} has no distogram_nll column -- it looks like a run WITHOUT the loss.")
    paired = pair(off, on)
    if paired.empty:
        raise SystemExit("no (target, attempt, stage) is present in both arms")
    check_same_blueprints(paired, args.off, args.on)

    summary = summarize(paired)
    summary.to_csv(os.path.join(out_path, "distogram_ab_summary.csv"), index=False)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        for stage in sorted(summary.stage.unique(), reverse=True):  # stage_2 first
            print(f"\n=== {stage}: on - off, paired by blueprint ===")
            print(summary[summary.stage == stage].drop(columns="stage").to_string(index=False))
    print("\nreference / diagnostics:")
    print_null(args.off, args.on)
    print_nll(args.on)

    if args.stage in set(paired.stage):
        print(f"\nfigure : {fig_paired(paired, args.stage, out_path)}")
    print(f"summary: {os.path.join(out_path, 'distogram_ab_summary.csv')}")


if __name__ == "__main__":
    main()
