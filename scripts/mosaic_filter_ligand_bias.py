"""Apply mosaic_scan.py's success filter to an EXISTING ligand_bias run.

New ligand_bias runs score every attempt on the fly (columns mosaic_*, passed; success.csv;
success/*.pdb). Runs made before that have only the final PDBs, which do not carry the PAE
matrices that ipSAE and the directional PAEs need. Two modes:

  --mode approx   (CPU, instant)  proxy filter from columns already in results.csv:
                  binder pLDDT > 0.8, hallucination-end ProteinMPNN recovery > 0.6 and mean
                  interface PAE < 5 A. ipSAE is NOT available and the two directional PAEs are
                  only available as their mean, so this is a necessary-condition screen: an
                  attempt failing it would fail mosaic's filter; passing it does not prove a pass.

  --mode exact    (needs a GPU, ~one forward pass per attempt) re-predicts each saved hardened
                  binder_sequence with mosaic's own scoring setup (single sample, 4 recycles,
                  25 sampling steps), computes mosaic's metrics() (ipsae, binder_target_pae,
                  target_binder_pae, plddt, recovery) and applies _passing(). The run's soft
                  final sequence was not saved, so recovery uses the one-hot hardened sequence
                  (ligand_bias.py itself uses the soft one, as mosaic does): a slightly stricter
                  recovery. Passing structures go to <run>/success_rescored/.

Writes <out_path>/mosaic_filter_<mode>.csv (one row per ligand/DNA attempt) and prints the pass
counts per target.

Usage:
    python3 scripts/mosaic_filter_ligand_bias.py --run_path runs/ligand_hallucination_pmpnninit
    python3 scripts/mosaic_filter_ligand_bias.py --run_path runs/ligand_hallucination_pmpnninit --mode exact
"""

import os
import json
import argparse

import numpy as np
import pandas as pd

from plot_ligand_bias import _load_results_csv, _short_label   # same directory

# mosaic_scan.py's _passing()
PASS = dict(ipsae=0.75, plddt=0.80, binder_target_pae=5.0, target_binder_pae=5.0, recovery=0.60)


def load(run_path):
    with open(f"{run_path}/results.csv") as f:
        n_cols = len(f.readline().rstrip("\n").split(","))
    df = _load_results_csv(f"{run_path}/results.csv", n_cols=n_cols)
    return df[(df["status"] == "ok") & (df["category"] != "apo")].copy()


def approx(df):
    out = df[["row_index", "condition_name", "category", "attempt_index", "plddt",
              "halluc_recovery", "interface_pae"]].copy()
    out["interface_pae_A"] = out["interface_pae"] * 32.0
    out["approx_pass"] = ((out["plddt"] > PASS["plddt"]) & (out["halluc_recovery"] > PASS["recovery"])
                          & (out["interface_pae_A"] < PASS["binder_target_pae"]))
    return out.drop(columns="interface_pae")


def exact(run_path, df, args):
    import jax
    import jax.numpy as jnp
    import flexcraft.sequence.aa_codes as aas
    from flexcraft.sequence.mpnn import make_pmpnn, batch_mpnn
    from flexcraft.structure.boltz import Joltz2, JoltzSpec

    manifest = json.load(open(f"{run_path}/ligands.json"))
    options = manifest["options"]
    num_aa = manifest["num_aa"]
    entries = {c["condition_index"]: c["entry"] for c in manifest["conditions"]}
    num_recycle = args.num_recycle or options.get("num_recycle", 4)
    sampling_steps = args.num_sampling_steps or options.get("num_sampling_steps", 25)

    model = Joltz2(cache=args.boltz_path)
    predict = model.predictor(num_recycle=num_recycle, num_samples=1,
                              num_sampling_steps=sampling_steps)
    _pmpnn, pmpnn_params = make_pmpnn(args.pmpnn_params, eps=0.05, split_params=True)
    batched_pmpnn = jax.jit(batch_mpnn(_pmpnn))

    def add_target(spec, e):
        if e["target_kind"] == "ccd":
            return spec.add_ccd(e["ccd"])
        if e["target_kind"] == "smiles":
            return spec.add_smiles(e["smiles"])
        return getattr(spec, "add_" + e["target_kind"])(e["sequence"], num_repeats=e.get("num_repeats", 1))

    def n_target_chains(e):
        return e.get("num_repeats", 1) if e["target_kind"] in ("dna", "rna") else 1

    def metrics(binder_onehot, key, result, e):
        """mosaic_scan.py's metrics(), restricted to the terms the filter uses."""
        res_data = result.to_data(return_samples=True)
        pmpnn_input = res_data.update(aa=aas.translate(res_data["aa"], aas.AF2_CODE, aas.PMPNN_CODE))
        is_binder = jnp.zeros((pmpnn_input.aa.shape[0],), dtype=jnp.bool_).at[:num_aa].set(True)
        logits = batched_pmpnn(pmpnn_params, key, pmpnn_input.drop_aa(where=is_binder))["logits"]
        logits = aas.translate_onehot(logits, aas.PMPNN_CODE, aas.AF2_CODE)[..., :20]
        if not result.is_single_sample:
            logits = logits.mean(axis=0)
        logits = logits[:num_aa]
        logits = logits - logits.mean(axis=0)
        prob = jax.nn.softmax(logits.at[:, aas.AF2_CODE.index("C")].set(-1e6) / 0.1, axis=-1)
        plddt, pae = result.plddt, 32 * result.pae
        if not result.is_single_sample:
            plddt, pae = plddt.mean(axis=0), pae.mean(axis=0)
        return dict(
            mosaic_plddt=float(plddt[:num_aa].mean()),
            mosaic_recovery=float((binder_onehot * prob).sum(axis=-1).mean()),
            mosaic_binder_target_pae=float(pae[:num_aa, num_aa:].mean()),
            mosaic_target_binder_pae=float(pae[num_aa:, :num_aa].mean()),
            mosaic_ipsae=float(result.ipsae(chain_index=is_binder)))

    rows = []
    for _, r in df.iterrows():
        if args.only_rows and int(r["row_index"]) not in args.only_rows:
            continue
        i = int(r["row_index"])
        e = entries[int(r["condition_index"])]
        seq = r["binder_sequence"]
        onehot = jax.nn.one_hot(aas.encode(seq, aas.AF2_CODE), 20)
        spec = add_target(JoltzSpec().add_protein(seq), e)
        key = jax.random.fold_in(jax.random.key(args.seed), i)
        pred = predict(key, spec)
        m = metrics(onehot, key, pred.result, e)
        passed = (m["mosaic_ipsae"] > PASS["ipsae"] and m["mosaic_plddt"] > PASS["plddt"]
                  and m["mosaic_binder_target_pae"] < PASS["binder_target_pae"]
                  and m["mosaic_target_binder_pae"] < PASS["target_binder_pae"]
                  and m["mosaic_recovery"] > PASS["recovery"])
        if passed:
            os.makedirs(f"{run_path}/success_rescored", exist_ok=True)
            pred.save_pdb(f"{run_path}/success_rescored/{i:04d}_{r['condition_name']}_a"
                          f"{int(r['attempt_index'])}.pdb")
        rows.append(dict(row_index=i, condition_name=r["condition_name"], category=r["category"],
                         attempt_index=int(r["attempt_index"]), **m, exact_pass=bool(passed)))
        print(f"[{i:04d}] {r['condition_name']:30s} ipsae {m['mosaic_ipsae']:.2f} plddt "
              f"{m['mosaic_plddt']:.2f} pae {m['mosaic_binder_target_pae']:.1f}/"
              f"{m['mosaic_target_binder_pae']:.1f} rec {m['mosaic_recovery']:.2f} -> "
              f"{'PASS' if passed else 'fail'}", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_path", required=True)
    ap.add_argument("--out_path", default=None)
    ap.add_argument("--mode", choices=["approx", "exact"], default="approx")
    ap.add_argument("--boltz_path", default="./params/boltz/")
    ap.add_argument("--pmpnn_params", default="params/pmpnn/v_48_030.pkl")
    ap.add_argument("--num_recycle", type=int, default=0, help="0 = the run's own setting")
    ap.add_argument("--num_sampling_steps", type=int, default=0, help="0 = the run's own setting")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only_rows", type=int, nargs="*", default=None, help="exact mode: row_index subset")
    args = ap.parse_args()
    out_path = args.out_path or args.run_path

    df = load(args.run_path)
    result = approx(df) if args.mode == "approx" else exact(args.run_path, df, args)
    flag = "approx_pass" if args.mode == "approx" else "exact_pass"
    result.to_csv(f"{out_path}/mosaic_filter_{args.mode}.csv", index=False)

    result["label"] = [_short_label(n, c) for n, c in zip(result["condition_name"], result["category"])]
    print(f"\n{args.mode} mosaic filter: {int(result[flag].sum())} of {len(result)} attempts pass")
    per_target = result.groupby("label", sort=False)[flag].agg(["sum", "size"])
    for label, r in per_target.iterrows():
        if r["sum"]:
            print(f"  {label:34s} {int(r['sum'])}/{int(r['size'])}")
    print(f"Wrote {out_path}/mosaic_filter_{args.mode}.csv")


if __name__ == "__main__":
    main()
