"""Break down mosaic_scan's _passing() filter gate by gate: of N designs, how many clear EACH
individual threshold, and how many clear all five together?

_passing() ANDs five conditions (flexcraft/pipelines/experimental/mosaic_scan.py:666-671):
    ipsae > 0.75, plddt > 0.8, binder_target_pae < 5.0, target_binder_pae < 5.0, recovery > 0.6
A single collapsed "passed" column (as in results.csv) can't show which gate is actually doing
the rejecting -- this script computes each gate separately from the same columns, so a gate that
blocks everything (e.g. ipsae for an AF2 run with a units bug, see flexcraft/structure/af/_data.py
:444-446) is visible on its own bar instead of being hidden inside one aggregate number.

Usage:
    python3 scripts/plot_distogram_passing_gates.py --run runs/mosaic_distogram_sweep_af2 --label af2
    python3 scripts/plot_distogram_passing_gates.py --run runs/mosaic/mosaic_distogram_sweep_joltz --label joltz
"""

import os
import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Mirrors flexcraft/pipelines/experimental/mosaic_scan.py's _passing() exactly -- mosaic_scan.py
# is a run-as-script pipeline (top-level argparse/model loading on import), not an importable
# module, so the condition is duplicated here rather than imported. Keep these two in sync.
GATES = [
    ("ipsae",              "ipSAE > 0.75",               lambda s: s > 0.75),
    ("plddt",              "pLDDT > 0.8",                lambda s: s > 0.8),
    ("binder_target_pae",  "binder→target pAE < 5.0 Å", lambda s: s < 5.0),
    ("target_binder_pae",  "target→binder pAE < 5.0 Å", lambda s: s < 5.0),
    ("recovery",           "self-consistency recovery > 0.6", lambda s: s > 0.6),
]

# Status palette (dataviz skill, references/palette.md) -- reserved colors for state, never
# reused as a categorical series color. Both clear 3:1 contrast on white.
GOOD, CRITICAL, INK, GREY = "#0ca30c", "#d03b3b", "#2B2B2B", "#8C8C8C"
PASS_THRESHOLD = 0.5  # >=50% of designs clearing a gate reads as "good", else "critical"


def load(path, label, stage):
    fp = os.path.join(path, "results.csv")
    if not os.path.isfile(fp):
        raise SystemExit(f"no results.csv in {path}")
    df = pd.read_csv(fp)
    df = df.drop_duplicates(subset=["target", "attempt", "stage", "distogram_weight"], keep="last")
    df = df[df.stage == stage]
    if df.empty:
        raise SystemExit(f"no {stage} rows in {fp}")
    if "structure_model" in df and not (df["structure_model"] == label).all():
        print(f"WARNING: {fp} structure_model={sorted(df['structure_model'].unique())}, not '{label}'")
    return df


def gate_table(df):
    rows = []
    joint = np.ones(len(df), dtype=bool)
    for col, cond_label, fn in GATES:
        s = pd.to_numeric(df[col], errors="coerce")
        ok = fn(s).fillna(False).to_numpy()
        joint &= ok
        rows.append(dict(gate=col, condition=cond_label, n_pass=int(ok.sum()), n=len(ok),
                         rate=ok.sum() / len(ok), median=float(s.median())))
    rows.append(dict(gate="ALL 5 GATES", condition="_passing() overall", n_pass=int(joint.sum()),
                     n=len(joint), rate=joint.sum() / len(joint), median=np.nan))
    table = pd.DataFrame(rows)
    # sanity check against the pipeline's own "passed" column, if present
    if "passed" in df:
        stored = (df["passed"].astype(str).str.strip() == "True").to_numpy()
        if not np.array_equal(stored, joint):
            n_mismatch = int((stored != joint).sum())
            print(f"WARNING: recomputed ALL-5-gates disagrees with the stored 'passed' column on "
                 f"{n_mismatch}/{len(df)} rows (recovery/ipsae ties or a mosaic_scan.py version "
                 f"drift from GATES above are the likely causes)")
    return table


def plot(table, label, stage, out_path):
    rows = table[table.gate != "ALL 5 GATES"].sort_values("rate")  # worst gate first
    overall = table[table.gate == "ALL 5 GATES"].iloc[0]

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    y = np.arange(len(rows))
    colors = [GOOD if r >= PASS_THRESHOLD else CRITICAL for r in rows["rate"]]
    ax.barh(y, rows["rate"] * 100, color=colors, height=0.6, zorder=3)
    for yi, (_, row) in zip(y, rows.iterrows()):
        ax.text(row["rate"] * 100 + 1.5, yi, f"{row['n_pass']}/{row['n']}  ({row['rate']*100:.0f}%)",
                va="center", ha="left", fontsize=9.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.condition}" for _, r in rows.iterrows()], fontsize=10)
    ax.set_xlim(0, 112)
    ax.set_xlabel("designs clearing this gate alone (%)", fontsize=10)
    ax.axvline(100, color=GREY, lw=1.0, ls="--", zorder=1)
    ax.grid(axis="x", color="#E6E6E6", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    ax.set_title(f"{label}: which _passing() gate is rejecting designs?  ({stage})",
                fontsize=12.5, fontweight="bold")
    fig.text(0.5, 0.02,
             f"Each bar is that ONE condition in isolation (not AND-ed with the others). "
             f"All five together: {overall['n_pass']}/{overall['n']} designs "
             f"({overall['rate']*100:.1f}%) end up in success/.",
             ha="center", va="bottom", fontsize=9, color="#555555", wrap=True)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    path = os.path.join(out_path, f"passing_gates_{label}.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="sweep run directory (has results.csv)")
    ap.add_argument("--label", required=True, help="short name used in the title/filename, e.g. af2")
    ap.add_argument("--stage", default="stage_2", help="stage to analyse (default: stage_2)")
    ap.add_argument("--out_path", default="runs/mosaic/mosaic_distogram_analysis",
                    help="where to write the PNG")
    args = ap.parse_args()
    os.makedirs(args.out_path, exist_ok=True)

    df = load(args.run, args.label, args.stage)
    table = gate_table(df)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 120):
        print(table.drop(columns="condition").to_string(index=False))
    fig_path = plot(table, args.label, args.stage, args.out_path)
    print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()
