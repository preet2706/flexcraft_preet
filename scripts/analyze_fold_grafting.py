"""Can a ligand be stuck into ANY fold?  Post-hoc analysis of a ligand_bias `salad_pmpnn` run.

In an `--init_mode salad_pmpnn` run every attempt pairs one arbitrary, target-unaware salad
blueprint (saved in <run>/blueprints/) with one hallucinated binder (<run>/predictions/). That
pairing is exactly the experiment "take a random fold, try to graft target X onto it", so
whether it worked can be computed afterwards, without rerunning anything:

  1. FOLD RETENTION  CA-RMSD (Kabsch) and DSSP identity of the final binder vs. its own
                     blueprint. A shuffled null (each binder vs. every OTHER attempt's blueprint)
                     calibrates what "same fold" means.
  2. ENGAGEMENT      the target is touched (contact_total) and confidently placed
                     (interface PAE), from results.csv.
  3. DESIGN QUALITY  binder pLDDT and ProteinMPNN self-consistency (hallucination-end recovery).
  4. GRAFTED         all four pass at once. Reported at a strict and a lenient threshold set.

Also: whether success depends on the fold (blueprint helix fraction / compactness), and, with
--compare_path (a random-init run), the blueprint-free criteria (2+3) side by side, to see what
the salad+ProteinMPNN start buys.

Writes into --out_path (default: the run folder):
    fold_grafting_per_attempt.csv   one row per attempt: RMSD/DSSP vs blueprint, criteria, tiers
    fold_grafting_summary.csv       one row per target: pass counts at both tiers
    fold_grafting_overview.png      null vs. matched RMSD, criteria funnel, per-attempt matrices
    fold_grafting_fold_dependence.png  outcomes vs. blueprint helix fraction / radius of gyration
    grafting_vs_random_init.png     (only with --compare_path) paired comparison of the two runs

Usage:
    python3 scripts/analyze_fold_grafting.py \\
        --run_path runs/ligand_hallucination_pmpnninit --compare_path runs/ligand_bias_hallucination

Thresholds (override with e.g. --strict "rmsd=2.5,plddt=0.85"; keys: rmsd [A], plddt, ipae [A],
contacts [# binder residues], recovery). ipae is stored normalized (PAE/32) in results.csv and is
converted back to Angstrom here.
"""

import os
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from flexcraft.utils import load_pdb
from plot_ligand_bias import (  # same directory
    _load_results_csv, _condition_order, _short_label,
    CATEGORY_COLOR, CATEGORY_LABEL, CATEGORY_ORDER, BLUE, GREY, INK)

STRICT = dict(rmsd=3.0, plddt=0.80, ipae=5.0, contacts=5, recovery=0.60)
LENIENT = dict(rmsd=5.0, plddt=0.70, ipae=10.0, contacts=3, recovery=0.50)
CRITERIA = ["retained", "engaged", "folded", "consistent"]
CRITERION_LABEL = {
    "retained": "fold\nretained",
    "engaged": "target\nengaged",
    "folded": "binder\nconfident",
    "consistent": "self-\nconsistent",
}


def _parse_thresholds(base: dict, text: str) -> dict:
    out = dict(base)
    for item in filter(None, (t.strip() for t in text.split(","))):
        key, value = item.split("=")
        assert key in out, f"unknown threshold {key!r}; expected one of {sorted(out)}"
        out[key] = float(value)
    return out


# ============================== structure helpers ==============================

def kabsch_rmsd(p: np.ndarray, q: np.ndarray) -> float:
    pc, qc = p - p.mean(0), q - q.mean(0)
    u, _, vt = np.linalg.svd(pc.T @ qc)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return float(np.sqrt((((pc @ rot.T) - qc) ** 2).sum(1).mean()))


def _binder(path: str, binder_only: bool):
    """(CA coordinates, DSSP int array, helix/strand/loop fractions) of chain 0."""
    data = load_pdb(path)
    if binder_only:
        data = data[np.asarray(data.chain_index) == 0]
    ca = np.asarray(data["atom_positions"])[:, 1]
    dssp = np.asarray(data.dssp)
    p = data.p_dssp
    return ca, dssp, dict(L=float(p["L"]), H=float(p["H"]), E=float(p["E"]))


def _rg(ca: np.ndarray) -> float:
    return float(np.sqrt(((ca - ca.mean(0)) ** 2).sum(1).mean()))


# ============================== criteria ==============================

def apply_criteria(df: pd.DataFrame, t: dict, tag: str, use_retained: bool = True) -> pd.DataFrame:
    """Adds boolean <criterion>_<tag> columns and grafted_<tag> (all criteria at once)."""
    out = df.copy()
    ipae_a = out["interface_pae"] * 32.0
    if use_retained:
        out[f"retained_{tag}"] = out["rmsd_to_blueprint"] <= t["rmsd"]
    out[f"engaged_{tag}"] = (out["contact_total"] >= t["contacts"]) & (ipae_a <= t["ipae"])
    out[f"folded_{tag}"] = out["plddt"] >= t["plddt"]
    out[f"consistent_{tag}"] = out["halluc_recovery"] >= t["recovery"]
    used = [c for c in CRITERIA if c != "retained" or use_retained]
    out[f"grafted_{tag}"] = np.all([out[f"{c}_{tag}"] for c in used], axis=0)
    return out


def _is_ligand_row(df):
    return (df["category"] != "apo").to_numpy()


# ============================== figures ==============================

def fig_overview(df, null, order_labels, order_cats, n_att, out_path, strict, lenient):
    fig = plt.figure(figsize=(17, 11.5))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 2.4], hspace=0.38, wspace=0.55)

    # (a) matched vs shuffled-null RMSD
    ax = fig.add_subplot(gs[0, 0:2])
    lig = df[_is_ligand_row(df)]
    bins = np.linspace(0, max(null.max(), lig["rmsd_to_blueprint"].max()) + 1, 40)
    ax.hist(null, bins=bins, density=True, color=GREY, alpha=0.55,
            label=f"null: binder vs. other attempts' blueprints (n={len(null)})")
    ax.hist(lig["rmsd_to_blueprint"], bins=bins, density=True, color=BLUE, alpha=0.75,
            label=f"matched: binder vs. its own blueprint (n={len(lig)})")
    for t, ls in ((strict, "-"), (lenient, "--")):
        ax.axvline(t["rmsd"], color=INK, ls=ls, lw=1.0)
    apo = df[df["category"] == "apo"]
    if len(apo):
        ax.axvline(float(apo["rmsd_to_blueprint"].iloc[0]), color="#C44E52", lw=1.6,
                   label="apo (no target) control")
    ax.set_xlabel("CA RMSD, binder vs. blueprint (Å)")
    ax.set_ylabel("density")
    ax.set_title("Does the binder keep its salad fold?", fontsize=10)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.55)
    ax.legend(frameon=False, fontsize=7)

    # (b) criteria funnel
    ax = fig.add_subplot(gs[0, 2:4])
    n = len(lig)
    width = 0.38
    for k, (tag, color) in enumerate((("strict", BLUE), ("lenient", "#9DB4D8"))):
        counts, ok = [], np.ones(n, dtype=bool)
        for c in CRITERIA:
            ok = ok & lig[f"{c}_{tag}"].to_numpy()
            counts.append(int(ok.sum()))
        bars = ax.bar(np.arange(len(CRITERIA)) + (k - 0.5) * width, counts, width, color=color,
                      label=f"{tag} thresholds")
        for b, c in zip(bars, counts):
            ax.text(b.get_x() + b.get_width() / 2, c + 0.8, str(c), ha="center", fontsize=8)
        single = [int(lig[f"{c}_{tag}"].sum()) for c in CRITERIA]
        ax.scatter(np.arange(len(CRITERIA)) + (k - 0.5) * width, single, marker="_", s=160,
                   color=INK, zorder=3, label="each criterion alone" if k == 0 else None)
    ax.set_xticks(range(len(CRITERIA)))
    ax.set_xticklabels([CRITERION_LABEL[c] for c in CRITERIA], fontsize=8)
    ax.set_ylabel(f"# of {n} ligand/DNA attempts passing")
    ax.set_title("Criteria funnel (bars) vs. each alone (ticks)", fontsize=10)
    ax.legend(frameon=False, fontsize=7)

    # (c) threshold sensitivity: grafted fraction vs. RMSD / interface-PAE cutoffs, with the
    # remaining criteria held at their lenient values
    ax = fig.add_subplot(gs[0, 4:6])
    rmsd_cuts, ipae_cuts = [2, 3, 4, 5, 7, 10], [5, 7.5, 10, 12.5, 15]
    sens = np.array([[np.mean((lig["rmsd_to_blueprint"] <= r) & (lig["ipae_A"] <= p)
                              & (lig["contact_total"] >= lenient["contacts"])
                              & (lig["plddt"] >= lenient["plddt"])
                              & (lig["halluc_recovery"] >= lenient["recovery"]))
                      for p in ipae_cuts] for r in rmsd_cuts])
    im = ax.imshow(sens, cmap="Blues", vmin=0, vmax=max(0.5, sens.max()), aspect="auto")
    for (i, j), v in np.ndenumerate(sens):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                color="white" if v > 0.6 * sens.max() else INK)
    ax.set_xticks(range(len(ipae_cuts)))
    ax.set_xticklabels(ipae_cuts, fontsize=8)
    ax.set_yticks(range(len(rmsd_cuts)))
    ax.set_yticklabels(rmsd_cuts, fontsize=8)
    ax.set_xlabel("interface PAE cutoff (Å)")
    ax.set_ylabel("RMSD-to-blueprint cutoff (Å)")
    ax.set_title("Grafted fraction vs. cutoffs\n(other criteria lenient)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)

    # (d-f) per-attempt matrices: ordinal 0 fail / 1 lenient / 2 strict
    cmap = ListedColormap(["#EDEDED", "#9DB4D8", BLUE])
    names = ["retained", "engaged", "grafted"]
    titles = {"retained": "fold retained", "engaged": "target engaged",
              "grafted": "GRAFTED (all criteria)"}
    labels = order_labels
    for k, name in enumerate(names):
        ax = fig.add_subplot(gs[1, 2 * k:2 * k + 2])
        mat = np.zeros((len(labels), n_att))
        for i, label in enumerate(labels):
            sub = df[df["_label"] == label].sort_values("attempt_index")
            for _, r in sub.iterrows():
                j = int(r["attempt_index"])
                mat[i, j] = 2 if r[f"{name}_strict"] else (1 if r[f"{name}_lenient"] else 0)
        ax.imshow(mat, cmap=cmap, vmin=0, vmax=2, aspect="auto", interpolation="nearest")
        if name == "retained":
            for i, label in enumerate(labels):
                sub = df[df["_label"] == label].sort_values("attempt_index")
                for _, r in sub.iterrows():
                    ax.text(int(r["attempt_index"]), i, f"{r['rmsd_to_blueprint']:.1f}",
                            ha="center", va="center", fontsize=5.5,
                            color="white" if mat[i, int(r["attempt_index"])] == 2 else INK)
        ax.set_xticks(range(n_att))
        ax.set_xticklabels([f"fold {j + 1}" for j in range(n_att)], fontsize=7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels if k == 0 else [], fontsize=7)
        if k == 0:
            for tick, cat in zip(ax.get_yticklabels(), order_cats):
                tick.set_color(CATEGORY_COLOR[cat])
        ax.set_title(titles[name], fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#EDEDED", "#9DB4D8", BLUE)]
    fig.legend(handles, ["fails", "passes lenient", "passes strict"], loc="lower center",
               ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Fold grafting: one arbitrary salad fold per column, one target per row "
                 "(retained cells show RMSD to blueprint in Å)", y=0.97)
    fig.savefig(f"{out_path}/fold_grafting_overview.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_fold_dependence(df, out_path):
    lig = df[_is_ligand_row(df)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    xs = [("blueprint_H", "blueprint helix fraction"), ("blueprint_rg", "blueprint radius of gyration (Å)")]
    ys = [("contact_total", "# binder residues contacting target"),
          ("ipae_A", "interface PAE (Å)"),
          ("rmsd_to_blueprint", "CA RMSD to blueprint (Å)")]
    for r, (xc, xl) in enumerate(xs):
        for c, (yc, yl) in enumerate(ys):
            ax = axes[r, c]
            for cat in CATEGORY_ORDER[1:]:
                sub = lig[lig["category"] == cat]
                ax.scatter(sub[xc], sub[yc], s=22, color=CATEGORY_COLOR[cat], alpha=0.8,
                           linewidths=0, label=CATEGORY_LABEL[cat])
            rho, p = spearmanr(lig[xc], lig[yc])
            # the 5 folds of one target share that target's overall level, so also test the
            # fold effect within targets: demean the outcome per target, keep the fold property
            y_within = lig[yc] - lig.groupby("condition_name")[yc].transform("mean")
            rho_w, p_w = spearmanr(lig[xc], y_within)
            print(f"  {xc:14s} vs {yc:18s} pooled rho {rho:+.2f} (p={p:.2g}) | "
                  f"within-target rho {rho_w:+.2f} (p={p_w:.2g})")
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.set_title(f"pooled ρ = {rho:+.2f} (p = {p:.2g})\n"
                         f"within-target ρ = {rho_w:+.2f} (p = {p_w:.2g})", fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=7.5)
    fig.suptitle("Does grafting success depend on the fold? (one point per attempt)", y=1.0)
    fig.tight_layout()
    fig.savefig(f"{out_path}/fold_grafting_fold_dependence.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _kde(ax, values, color, label, fill=True, lw=1.8, ls="-"):
    """Silverman-bandwidth KDE (as scripts/plot_mosaic_scan.py), a vertical line if degenerate."""
    from scipy.stats import gaussian_kde
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return
    if v.size < 3 or np.isclose(v.std(), 0.0):
        ax.axvline(v.mean(), color=color, lw=lw, ls=ls, label=label)
        return
    pad = 0.05 * (v.max() - v.min() + 1e-9)
    grid = np.linspace(v.min() - pad, v.max() + pad, 512)
    dens = gaussian_kde(v, bw_method="silverman")(grid)
    ax.plot(grid, dens, color=color, lw=lw, ls=ls, label=label)
    if fill:
        ax.fill_between(grid, dens, color=color, alpha=0.18)


def fig_fold_recovery(df, null_rmsd, null_dssp, order_labels, order_cats, out_path):
    """Per-ligand fold recovery: CA RMSD and DSSP identity to the blueprint, as density curves
    against the shuffled null (as in the mosaic plots) and as per-ligand means over attempts."""
    lig = df[_is_ligand_row(df)]
    apo = df[df["category"] == "apo"]
    has_init = "rmsd_init_to_blueprint" in df.columns
    metrics = [("rmsd_to_blueprint", "rmsd_init_to_blueprint", null_rmsd,
                "CA RMSD to blueprint (Å)", "lower = fold recovered"),
               ("dssp_identity_to_blueprint", "dssp_identity_init_to_blueprint", null_dssp,
                "DSSP identity to blueprint (fraction of residues)", "higher = fold recovered")]
    n = len(order_labels) - 1                       # drop the apo row from the per-ligand panels
    labels = [l for l in order_labels if l != "apo"]
    cats = [c for c in order_cats if c != "apo"]
    fig = plt.figure(figsize=(14, 0.32 * n + 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, max(1.0, n * 0.11)], hspace=0.3, wspace=0.12)
    for k, (col, init_col, null, xlabel, note) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, k])
        _kde(ax, null, GREY, f"shuffled null (n={len(null)})")
        for cat in CATEGORY_ORDER[1:]:
            sub = lig[lig["category"] == cat]
            if len(sub):
                _kde(ax, sub[col], CATEGORY_COLOR[cat], f"{CATEGORY_LABEL[cat]} (n={len(sub)})",
                     fill=False)
        if has_init:
            _kde(ax, lig[init_col], INK, "initial sequence, before hallucination", fill=False, ls="--")
        if len(apo):
            ax.axvline(float(apo[col].iloc[0]), color="#C44E52", lw=1.5, label="apo (no target)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density" if k == 0 else "")
        ax.set_title(f"Density, all attempts ({note})", fontsize=10)
        if k == 0:
            ax.legend(frameon=False, fontsize=7)

        # per-ligand: every attempt, with mean +/- SD over the attempts
        ax = fig.add_subplot(gs[1, k])
        y = np.arange(n)
        rng = np.random.default_rng(0)
        ax.axvline(np.median(null), color=GREY, ls="--", lw=1.2, zorder=0)
        if len(apo):
            ax.axvline(float(apo[col].iloc[0]), color="#C44E52", lw=1.2, zorder=0)
        for yi, label, cat in zip(y, labels, cats):
            vals = lig.loc[lig["_label"] == label, col].to_numpy()
            ax.scatter(vals, yi + rng.uniform(-0.18, 0.18, len(vals)), s=15,
                       color=CATEGORY_COLOR[cat], alpha=0.75, linewidths=0, zorder=3)
            ax.errorbar(vals.mean(), yi, xerr=vals.std(ddof=1) if len(vals) > 1 else 0, fmt="D",
                        color=INK, ms=5, capsize=2.5, elinewidth=1.2, zorder=4)
            if has_init:
                iv = lig.loc[lig["_label"] == label, init_col]
                ax.scatter([iv.mean()], [yi], marker="^", s=34, facecolors="white",
                           edgecolors=INK, linewidths=1.0, zorder=5)
        ax.set_yticks(y)
        ax.set_yticklabels(labels if k == 0 else [], fontsize=8)
        if k == 0:
            for tick, cat in zip(ax.get_yticklabels(), cats):
                tick.set_color(CATEGORY_COLOR[cat])
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title("Per ligand: mean ± SD (diamond) of the attempts (dots)", fontsize=10)
    if has_init:
        fig.axes[-1].scatter([], [], marker="^", s=34, facecolors="white", edgecolors=INK,
                             label="initial structure (mean)")
        fig.axes[-1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("Fold recovery of the salad blueprint (grey dashed = shuffled-null median, "
                 "red = apo control)", y=0.94)
    fig.savefig(f"{out_path}/fold_recovery.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_vs_random(per_target, out_path, run_label, compare_label):
    fig = plt.figure(figsize=(15, 8.6))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)
    panel_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
                  fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 0])]
    panels = [("contact_total", "interface size (# residues)"),
              ("ipae_A", "interface PAE (Å)"), ("plddt", "binder pLDDT"),
              ("halluc_recovery", "ProteinMPNN recovery")]
    for ax, (col, label) in zip(panel_axes, panels):
        for cat in CATEGORY_ORDER[1:]:
            sub = per_target[per_target["category"] == cat]
            ax.scatter(sub[f"{col}_random"], sub[f"{col}_pmpnn"], s=30, color=CATEGORY_COLOR[cat],
                       linewidths=0, label=CATEGORY_LABEL[cat])
        lo = min(per_target[f"{col}_random"].min(), per_target[f"{col}_pmpnn"].min())
        hi = max(per_target[f"{col}_random"].max(), per_target[f"{col}_pmpnn"].max())
        ax.plot([lo, hi], [lo, hi], color=GREY, ls="--", lw=1, zorder=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = wilcoxon(per_target[f"{col}_pmpnn"], per_target[f"{col}_random"]).pvalue
        ax.set_xlabel(f"{compare_label} (per-target mean)")
        ax.set_ylabel(f"{run_label} (per-target mean)")
        delta = float((per_target[f"{col}_pmpnn"] - per_target[f"{col}_random"]).median())
        print(f"  {col:16s} median per-target difference (pmpnn - random) {delta:+.3f}, "
              f"Wilcoxon p = {p:.3g}")
        ax.set_title(f"{label}\nWilcoxon across {len(per_target)} targets: p = {p:.2g}", fontsize=9.5)
    panel_axes[0].legend(frameon=False, fontsize=7.5)

    # lenient grafting-ready rate per target: dumbbell between the two inits
    ax = fig.add_subplot(gs[1, 1:3])
    x = np.arange(len(per_target))
    rnd, pm = per_target["rate_random_lenient"].to_numpy(), per_target["rate_pmpnn_lenient"].to_numpy()
    ax.vlines(x, np.minimum(rnd, pm), np.maximum(rnd, pm), color=GREY, lw=1.2, zorder=1)
    ax.scatter(x, rnd, s=34, facecolors="white", edgecolors=INK, zorder=2, label=compare_label)
    ax.scatter(x, pm, s=34, color=BLUE, zorder=3, label=run_label)
    ax.set_xticks(x)
    ax.set_xticklabels(per_target["label"], rotation=75, ha="right", fontsize=7)
    for tick, cat in zip(ax.get_xticklabels(), per_target["category"]):
        tick.set_color(CATEGORY_COLOR[cat])
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("fraction of attempts")
    strict_pm = per_target["rate_pmpnn_strict"].mean()
    strict_rn = per_target["rate_random_strict"].mean()
    ax.set_title("Blueprint-free grafting-ready rate, lenient (engaged + confident + "
                 f"self-consistent)\nstrict rate, mean over targets: {run_label} {strict_pm:.2f} "
                 f"vs. {compare_label} {strict_rn:.2f}", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"What does the salad+ProteinMPNN start buy? ({run_label} vs. {compare_label})", y=0.97)
    fig.savefig(f"{out_path}/grafting_vs_random_init.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ============================== main ==============================

def _load_run(path):
    with open(f"{path}/results.csv") as f:
        n_cols = len(f.readline().rstrip("\n").split(","))
    df = _load_results_csv(f"{path}/results.csv", n_cols=n_cols)
    df = df[df["status"] == "ok"].copy()
    df["ipae_A"] = df["interface_pae"] * 32.0
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_path", required=True, help="a salad_pmpnn ligand_bias run folder")
    ap.add_argument("--compare_path", default=None, help="a random-init ligand_bias run folder")
    ap.add_argument("--out_path", default=None)
    ap.add_argument("--strict", default="", help='override strict thresholds, e.g. "rmsd=2.5,plddt=0.85"')
    ap.add_argument("--lenient", default="", help="override lenient thresholds")
    args = ap.parse_args()
    out_path = args.out_path or args.run_path
    strict, lenient = _parse_thresholds(STRICT, args.strict), _parse_thresholds(LENIENT, args.lenient)

    import json
    manifest = json.load(open(f"{args.run_path}/ligands.json"))
    assert manifest["options"].get("init_mode") == "salad_pmpnn", (
        f"{args.run_path} is not a salad_pmpnn run (init_mode="
        f"{manifest['options'].get('init_mode')!r}): no blueprints to compare against.")
    n_att = int(manifest["options"]["attempts_per_target"])
    df = _load_run(args.run_path)

    # ---- fold retention vs. the attempt's own blueprint, plus the shuffled null ----
    bp_ca, pred_ca, bp_dssp, pred_dssp = {}, {}, {}, {}
    records = []
    for _, r in df.iterrows():
        i, name, a = int(r["row_index"]), r["condition_name"], int(r["attempt_index"])
        stem = f"{i:04d}_{name}_a{a}.pdb"
        b_ca, b_dssp, b_comp = _binder(f"{args.run_path}/blueprints/{stem}", binder_only=False)
        p_ca, p_dssp, _ = _binder(f"{args.run_path}/predictions/{stem}", binder_only=True)
        bp_ca[i], pred_ca[i], bp_dssp[i], pred_dssp[i] = b_ca, p_ca, b_dssp, p_dssp
        rec = dict(
            row_index=i, rmsd_to_blueprint=kabsch_rmsd(p_ca, b_ca),
            dssp_identity_to_blueprint=float((b_dssp == p_dssp).mean()),
            blueprint_H=b_comp["H"], blueprint_E=b_comp["E"], blueprint_L=b_comp["L"],
            blueprint_rg=_rg(b_ca))
        # runs made with --save_init True also saved Boltz's structure for the initial
        # (salad -> ProteinMPNN) sequence: does Boltz reproduce the blueprint before any
        # hallucination, and how far does hallucination then move it?
        init_file = f"{args.run_path}/init/{stem[:-4]}_init.pdb"
        if os.path.exists(init_file):
            i_ca, i_dssp, _ = _binder(init_file, binder_only=True)
            rec.update(rmsd_init_to_blueprint=kabsch_rmsd(i_ca, b_ca),
                       dssp_identity_init_to_blueprint=float((b_dssp == i_dssp).mean()),
                       rmsd_final_to_init=kabsch_rmsd(p_ca, i_ca))
        records.append(rec)
    df = df.merge(pd.DataFrame(records), on="row_index")
    rows = list(bp_ca)
    null = np.array([kabsch_rmsd(pred_ca[i], bp_ca[j]) for i in rows for j in rows if i != j])
    null_dssp = np.array([float((pred_dssp[i] == bp_dssp[j]).mean())
                          for i in rows for j in rows if i != j])

    df = apply_criteria(df, strict, "strict")
    df = apply_criteria(df, lenient, "lenient")
    ordered = _condition_order(manifest["conditions"])
    label_of = {c["condition_index"]: ("apo" if c["entry"] is None else
                _short_label(c["entry"]["name"], c["entry"]["category"])) for c in ordered}
    cat_of = {c["condition_index"]: ("apo" if c["entry"] is None else c["entry"]["category"])
              for c in ordered}
    df["_label"] = df["condition_index"].map(label_of)
    order_labels = [label_of[c["condition_index"]] for c in ordered]
    order_cats = [cat_of[c["condition_index"]] for c in ordered]

    cols = ["row_index", "condition_index", "attempt_index", "condition_name", "category",
            "rmsd_to_blueprint", "dssp_identity_to_blueprint", "blueprint_H", "blueprint_E",
            "blueprint_rg", "contact_total", "ipae_A", "plddt", "halluc_recovery"] + \
           [f"{c}_{t}" for t in ("strict", "lenient") for c in CRITERIA + ["grafted"]]
    df[cols].to_csv(f"{out_path}/fold_grafting_per_attempt.csv", index=False)

    # ---- per-target summary ----
    lig = df[_is_ligand_row(df)]
    agg = {f"n_{c}_{t}": (f"{c}_{t}", "sum") for t in ("strict", "lenient")
           for c in CRITERIA + ["grafted"]}
    summary = lig.groupby("condition_index").agg(
        condition_name=("condition_name", "first"), category=("category", "first"),
        n_attempts=("row_index", "size"),
        rmsd_to_blueprint_mean=("rmsd_to_blueprint", "mean"),
        rmsd_to_blueprint_std=("rmsd_to_blueprint", "std"),
        rmsd_to_blueprint_min=("rmsd_to_blueprint", "min"),
        dssp_identity_mean=("dssp_identity_to_blueprint", "mean"),
        dssp_identity_std=("dssp_identity_to_blueprint", "std"), **agg)
    summary = summary.loc[[c["condition_index"] for c in ordered if c["entry"] is not None]]
    summary.to_csv(f"{out_path}/fold_grafting_summary.csv")

    # ---- console report ----
    print(f"strict thresholds : {strict}\nlenient thresholds: {lenient}\n")
    print(f"fold retention: matched RMSD median {lig.rmsd_to_blueprint.median():.2f} A "
          f"(IQR {lig.rmsd_to_blueprint.quantile(.25):.2f}-{lig.rmsd_to_blueprint.quantile(.75):.2f}) "
          f"vs. shuffled null median {np.median(null):.2f} A "
          f"(5th percentile {np.percentile(null, 5):.2f} A)")
    apo = df[df["category"] == "apo"]
    if len(apo):
        print(f"apo control (no target) RMSD to its blueprint: {float(apo.rmsd_to_blueprint.iloc[0]):.2f} A")
    print(f"\nattempts passing, out of {len(lig)} ligand/DNA attempts:")
    print(f"  {'criterion':12s} {'strict':>8s} {'lenient':>8s}")
    for c in CRITERIA + ["grafted"]:
        print(f"  {c:12s} {int(lig[c + '_strict'].sum()):8d} {int(lig[c + '_lenient'].sum()):8d}")
    print("\nper category (attempts grafted, strict / lenient, of n):")
    for cat in CATEGORY_ORDER[1:]:
        sub = lig[lig["category"] == cat]
        if len(sub):
            print(f"  {cat:4s} {int(sub.grafted_strict.sum()):3d} / {int(sub.grafted_lenient.sum()):3d}  of {len(sub)}")
    print("\nper target (grafted strict/lenient of n attempts; folds retained strict/lenient):")
    for _, r in summary.iterrows():
        print(f"  {r.condition_name:34s} {int(r.n_grafted_strict)}/{int(r.n_grafted_lenient)} of "
              f"{int(r.n_attempts)}   retained {int(r.n_retained_strict)}/{int(r.n_retained_lenient)}   "
              f"mean RMSD {r.rmsd_to_blueprint_mean:.2f} A")

    fig_overview(df, null, order_labels, order_cats, n_att, out_path, strict, lenient)
    fig_fold_dependence(df, out_path)
    fig_fold_recovery(df, null, null_dssp, order_labels, order_cats, out_path)
    print(f"\nWrote fold_grafting_{{per_attempt,summary}}.csv, fold_grafting_overview.png, "
          f"fold_grafting_fold_dependence.png, fold_recovery.png to {out_path}")

    # ---- comparison with the random-init run (blueprint-free criteria only) ----
    if args.compare_path:
        cmp_manifest = json.load(open(f"{args.compare_path}/ligands.json"))
        assert cmp_manifest["options"].get("init_mode") == "random", \
            f"--compare_path should be a random-init run, got {cmp_manifest['options'].get('init_mode')!r}"
        rnd = _load_run(args.compare_path)
        rnd = apply_criteria(rnd, strict, "strict", use_retained=False)
        rnd = apply_criteria(rnd, lenient, "lenient", use_retained=False)
        pm = apply_criteria(lig.copy(), strict, "strict", use_retained=False)
        pm = apply_criteria(pm, lenient, "lenient", use_retained=False)
        rnd = rnd[_is_ligand_row(rnd)]
        metrics = ["contact_total", "ipae_A", "plddt", "halluc_recovery"]
        keep = ["condition_name", "category"]
        g_pm = pm.groupby("condition_name")
        g_rn = rnd.groupby("condition_name")
        per_target = pd.DataFrame({
            "category": g_pm["category"].first(),
            **{f"{m}_pmpnn": g_pm[m].mean() for m in metrics},
            **{f"{m}_random": g_rn[m].mean() for m in metrics},
            "rate_pmpnn_strict": g_pm["grafted_strict"].mean(),
            "rate_random_strict": g_rn["grafted_strict"].mean(),
            "rate_pmpnn_lenient": g_pm["grafted_lenient"].mean(),
            "rate_random_lenient": g_rn["grafted_lenient"].mean()}).dropna()
        per_target["label"] = [_short_label(n, c) for n, c in zip(per_target.index, per_target["category"])]
        per_target = per_target.sort_values("category", key=lambda s: s.map(CATEGORY_ORDER.index))
        per_target.to_csv(f"{out_path}/grafting_vs_random_init.csv")
        print("\nblueprint-free grafting-ready rate (engaged + confident + self-consistent):")
        print(f"  {'':10s} {'salad+pmpnn':>12s} {'random':>8s}")
        for tag in ("strict", "lenient"):
            print(f"  {tag:10s} {pm['grafted_' + tag].mean():12.2f} {rnd['grafted_' + tag].mean():8.2f}")
        fig_vs_random(per_target, out_path, f"{manifest['options']['init_mode']} init",
                      "random init")
        print(f"Wrote grafting_vs_random_init.png/csv to {out_path}")


if __name__ == "__main__":
    main()
