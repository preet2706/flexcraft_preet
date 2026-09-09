"""Compare the interface residues of the docked blueprint against the hallucinated complex.

Answers "did the binder land where the blueprint put it?" at per-residue resolution, without
opening PyMOL. Reads the PDBs a docked mosaic_scan run wrote and emits:

  epitope_residues.csv   long format: one row per (attempt, stage, target residue)
  epitope_summary.txt    human-readable residue lists per attempt
  epitope_map.png        per-residue map + how often each residue is used, blueprint vs predicted

Usage:
    python3 scripts/epitope_report.py --run_path runs/mosaic_scan_docked_L80/

Chain layout is detected from the sequence, not the chain letter, because the two file kinds
disagree: blueprints are written TARGET-first (salad's parse_target layout) and attempts
BINDER-first (Boltz's layout).
"""

import os
import re
import glob
import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from flexcraft.utils import load_pdb
from flexcraft.files import FastaFile
import flexcraft.sequence.aa_codes as aas

# House palette, matching plot_mosaic_scan.py
BLUE, ORANGE, GREEN, GREY = "#4C72B0", "#DD8452", "#55A868", "#8C8C8C"
C_NEITHER, C_BP, C_PRED, C_SHARED = "#F2F2F2", BLUE, ORANGE, GREEN

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
})


def split_complex(data, target_sequence):
    """Return (binder, target) DesignData, identifying the target by SEQUENCE.

    Robust to the blueprint/attempt chain-order inversion and to chain relabelling.
    """
    chains = sorted(set(np.asarray(data.chain_index).tolist()))
    assert len(chains) == 2, f"expected a 2-chain complex, got chains {chains}"
    parts = {c: data[np.asarray(data.chain_index) == c] for c in chains}
    seqs = {c: p.to_sequence_string() for c, p in parts.items()}
    hits = [c for c in chains if seqs[c] == target_sequence]
    if len(hits) != 1:
        raise AssertionError(
            "could not identify the target chain by sequence.\n"
            f"  wanted ({len(target_sequence)}): {target_sequence}\n"
            + "".join(f"  chain {c} ({len(seqs[c])}): {seqs[c]}\n" for c in chains))
    t = hits[0]
    b = [c for c in chains if c != t][0]
    return parts[b], parts[t]


def contact_profile(binder, target, cutoff):
    """Per-target-residue: (in-contact mask, distance to the nearest binder residue).

    Pseudo-CB, matching the pipeline's epitope_mask (mosaic_scan.py) and Boltz's own contact
    atom. Pseudo-CB is rebuilt from N/CA/C, so it does not depend on generated sidechains.
    """
    cb_b = np.asarray(binder.cb)
    cb_t = np.asarray(target.cb)
    d = np.linalg.norm(cb_t[:, None] - cb_b[None, :], axis=-1).min(axis=1)
    return d < cutoff, d


def jaccard(a, b):
    u = float((a | b).sum())
    return float((a & b).sum()) / u if u else 0.0


def parse_hotspot_spec(spec, target_length):
    """Expand the compact 'A8+A44+A20-30' form written into results.csv."""
    mask = np.zeros(target_length, bool)
    if not isinstance(spec, str) or spec in ("none", "nan", ""):
        return mask
    for tok in spec.split("+"):
        tok = tok.strip()
        if not tok:
            continue
        rest = tok[1:]
        if "-" in rest:
            a, b = [int(v) for v in rest.split("-", 1)]
        else:
            a = b = int(rest)
        mask[a - 1:b] = True          # spec is 1-based over the fasta sequence
    return mask


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_path", default="runs/mosaic_scan_docked_L80/")
    ap.add_argument("--targets", default="runs/targets.fasta")
    ap.add_argument("--out_path", default=None, help="defaults to --run_path")
    ap.add_argument("--cutoff", type=float, default=8.0)
    args = ap.parse_args()
    out_path = args.out_path or args.run_path
    os.makedirs(out_path, exist_ok=True)

    fasta = {k: v.strip().upper() for k, v in FastaFile(args.targets).items()}
    results = pd.read_csv(os.path.join(args.run_path, "results.csv"))
    hotspot_by_target = (results.groupby("target").hotspot_spec.first().to_dict()
                         if "hotspot_spec" in results else {})

    rows, records = [], []
    for bp_path in sorted(glob.glob(os.path.join(args.run_path, "blueprints", "*_complex.pdb"))):
        m = re.match(r"(.+)_(\d+)_complex\.pdb$", os.path.basename(bp_path))
        if not m:
            continue
        target_name, attempt = m.group(1), int(m.group(2))
        if target_name not in fasta:
            print(f"skip {bp_path}: '{target_name}' not in {args.targets}")
            continue
        tseq = fasta[target_name]

        bp_binder, bp_target = split_complex(load_pdb(bp_path), tseq)
        bp_mask, bp_dist = contact_profile(bp_binder, bp_target, args.cutoff)

        for stage in ("stage_1", "stage_2"):
            pred_path = os.path.join(args.run_path, "attempts",
                                     f"{target_name}_{attempt}_{stage}.pdb")
            if not os.path.isfile(pred_path):
                continue          # an interrupted run leaves the last attempt incomplete
            pr_binder, pr_target = split_complex(load_pdb(pred_path), tseq)
            pr_mask, pr_dist = contact_profile(pr_binder, pr_target, args.cutoff)

            hot = parse_hotspot_spec(hotspot_by_target.get(target_name), len(tseq))
            for i, aa in enumerate(tseq):
                rows.append(dict(
                    target=target_name, attempt=attempt, stage=stage,
                    resi=i + 1, aa=aa,
                    in_blueprint=bool(bp_mask[i]), in_predicted=bool(pr_mask[i]),
                    shared=bool(bp_mask[i] and pr_mask[i]),
                    in_hotspot=bool(hot[i]),
                    dist_blueprint=round(float(bp_dist[i]), 2),
                    dist_predicted=round(float(pr_dist[i]), 2)))
            records.append(dict(
                target=target_name, attempt=attempt, stage=stage,
                bp=bp_mask.copy(), pred=pr_mask.copy(), hot=hot, seq=tseq))

    assert records, f"no blueprint/attempt PDB pairs found under {args.run_path}"
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_path, "epitope_residues.csv")
    df.to_csv(csv_path, index=False)

    # ---- text summary ----------------------------------------------------------------
    def label(seq, mask):
        return " ".join(f"{seq[i]}{i+1}" for i in np.where(mask)[0]) or "(none)"

    lines = [f"Epitope comparison  (CB-CB cutoff {args.cutoff} A)",
             f"run: {args.run_path}", ""]
    for r in records:
        inter = r["bp"] & r["pred"]
        lines += [
            f"--- {r['target']} attempt {r['attempt']} {r['stage']} ---",
            f"  blueprint epitope ({int(r['bp'].sum()):2d}): {label(r['seq'], r['bp'])}",
            f"  predicted epitope ({int(r['pred'].sum()):2d}): {label(r['seq'], r['pred'])}",
            f"  shared            ({int(inter.sum()):2d}): {label(r['seq'], inter)}",
            f"  Jaccard {jaccard(r['bp'], r['pred']):.3f}   "
            f"recall(bp) {inter.sum()/max(1,r['bp'].sum()):.3f}", ""]

    # ---- matched vs shuffled null, computed here so an interrupted run still gets one ---
    by_target = {}
    for r in records:
        by_target.setdefault((r["target"], r["stage"]), []).append(r)
    matched, shuffled = [], []
    for key, group in by_target.items():
        for a in group:
            matched.append(jaccard(a["bp"], a["pred"]))
            for b in group:
                if a["attempt"] != b["attempt"]:
                    shuffled.append(jaccard(a["bp"], b["pred"]))
    lines += ["=== matched vs shuffled null ==="]
    lines += [f"  matched  n={len(matched):3d}  mean Jaccard {np.mean(matched):.3f}"]
    if shuffled:
        lines += [f"  shuffled n={len(shuffled):3d}  mean Jaccard {np.mean(shuffled):.3f}",
                  f"  delta (matched - shuffled) = {np.mean(matched)-np.mean(shuffled):+.3f}"]
    else:
        lines += ["  shuffled: not computable (need >1 attempt per target)"]
    txt = "\n".join(lines)
    summary_path = os.path.join(out_path, "epitope_summary.txt")
    with open(summary_path, "wt") as f:
        f.write(txt + "\n")
    print(txt)

    # ---- figure ------------------------------------------------------------------------
    target_name = records[0]["target"]
    recs = [r for r in records if r["target"] == target_name]
    seq = recs[0]["seq"]
    L = len(seq)

    fig, axes = plt.subplots(
        2, 1, figsize=(max(14, L * 0.19), 3.0 + 0.42 * len(recs)),
        gridspec_kw=dict(height_ratios=[max(1.6, 0.34 * len(recs)), 1.5], hspace=0.42))

    # panel 1: per-residue map, one row per (attempt, stage)
    ax = axes[0]
    grid = np.zeros((len(recs), L), int)     # 0 neither, 1 bp only, 2 pred only, 3 shared
    for i, r in enumerate(recs):
        grid[i] = r["bp"].astype(int) + 2 * r["pred"].astype(int)
    cmap = matplotlib.colors.ListedColormap([C_NEITHER, C_BP, C_PRED, C_SHARED])
    ax.imshow(grid, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=3,
              extent=[0.5, L + 0.5, len(recs) - 0.5, -0.5])
    ax.set_yticks(range(len(recs)))
    ax.set_yticklabels([f"att {r['attempt']} {r['stage'].replace('stage_','s')}" for r in recs],
                       fontsize=8)
    ax.set_xlabel(f"{target_name} residue")
    ax.set_title(f"Interface residues on {target_name}: blueprint vs hallucinated complex")
    for x in range(len(recs)):
        ax.axhline(x + 0.5, color="white", lw=1.2)

    # hotspot track under the map (data coords; no custom transform, which would defeat
    # tight_layout)
    hot = recs[0]["hot"]
    if hot.any():
        for i in np.where(hot)[0]:
            ax.add_patch(plt.Rectangle((i + 0.5, len(recs) - 0.42), 1, 0.28,
                                       color=GREY, clip_on=False, zorder=5))
        ax.set_ylim(len(recs) - 0.1, -0.5)
        ax.annotate("requested hotspot", xy=(0.0, -0.02), xycoords="axes fraction",
                    xytext=(0, -3), textcoords="offset points",
                    fontsize=7.5, color=GREY, va="top", ha="left")

    # panel 2: how often each residue is used
    ax = axes[1]
    bp_freq = np.mean([r["bp"] for r in recs], axis=0)
    pr_freq = np.mean([r["pred"] for r in recs], axis=0)
    x = np.arange(1, L + 1)
    ax.bar(x, bp_freq, color=BLUE, alpha=0.75, width=0.85, label="blueprint")
    ax.bar(x, -pr_freq, color=ORANGE, alpha=0.85, width=0.85, label="hallucinated")
    ax.axhline(0, color="#444444", lw=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticklabels(["1.0", "0.5", "0", "0.5", "1.0"])
    ax.set_ylabel("fraction of designs\nin contact")
    ax.set_xlabel(f"{target_name} residue")
    ax.set_xlim(0.5, L + 0.5)
    ax.set_title("Which residues each side uses (blueprint up, hallucinated down)")
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper right")

    handles = [Patch(color=C_BP, label="blueprint only"),
               Patch(color=C_PRED, label="hallucinated only"),
               Patch(color=C_SHARED, label="both (shared epitope)"),
               Patch(color=C_NEITHER, label="neither")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=4, frameon=False, fontsize=10)
    d_txt = (f"matched Jaccard {np.mean(matched):.3f}"
             + (f"   shuffled null {np.mean(shuffled):.3f}"
                f"   $\\Delta$ {np.mean(matched)-np.mean(shuffled):+.3f}" if shuffled else ""))
    fig.text(0.5, 0.005, f"CB-CB cutoff {args.cutoff} A.   {d_txt}",
             ha="center", fontsize=9, color="#666666")
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig_path = os.path.join(out_path, "epitope_map.png")
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    for p in (csv_path, summary_path, fig_path):
        print("wrote", p)


if __name__ == "__main__":
    main()
