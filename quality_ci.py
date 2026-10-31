#!/usr/bin/env python3
"""
Paired bootstrap confidence intervals for KV-cache compression quality results (S4).

Input: a CSV (or JSONL) with ONE ROW PER EVALUATED EXAMPLE:
    model,method,dataset,example_id,score
    llama-8b,FullKV,narrativeqa,0,0.3125
    llama-8b,KIVI2,narrativeqa,0,0.2857
    ...
- `score` is the per-example LongBench metric (F1 / ROUGE-L / edit sim, on 0-1 or 0-100)
  or 0/1 correctness for LongBench-v2.
- Optional column `group` (e.g. short/medium/long, or thinking/non-thinking) is treated
  as part of the dataset key, so Study B can be analysed per length group and mode.

Because decoding is greedy, the only uncertainty is sampling over benchmark instances,
which is exactly what a bootstrap over examples estimates. We use a PAIRED bootstrap:
the same resampled example indices are used for the method and for Full KV, so the CI
is on the difference and benefits from the strong per-example correlation.

Output: per (model, dataset[, group], method):
    mean, 95% CI of mean, delta vs baseline, 95% CI of delta, and whether the delta CI
    excludes zero. Also writes a LaTeX-ready "mean ± half-width" table.

Usage:
    python quality_ci.py scores.csv --baseline FullKV --scale 100 --out ci_results.csv
"""
import argparse
import sys

import numpy as np
import pandas as pd


def load(path: str) -> pd.DataFrame:
    if path.endswith(".jsonl"):
        df = pd.read_json(path, lines=True)
    else:
        df = pd.read_csv(path)
    required = {"model", "method", "dataset", "example_id", "score"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"missing columns: {sorted(missing)}")
    return df


def paired_bootstrap(base: np.ndarray, meth: np.ndarray, n_boot: int, rng, alpha: float):
    n = len(base)
    idx = rng.integers(0, n, size=(n_boot, n))
    m_boot = meth[idx].mean(axis=1)
    d_boot = (meth - base)[idx].mean(axis=1)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return (
        meth.mean(),
        np.percentile(m_boot, [lo, hi]),
        (meth - base).mean(),
        np.percentile(d_boot, [lo, hi]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores")
    ap.add_argument("--baseline", default="FullKV")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply scores by this (e.g. 100 if scores are 0-1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ci_results.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = load(args.scores)
    df["score"] = df["score"].astype(float) * args.scale
    keys = ["model", "dataset"] + (["group"] if "group" in df.columns else [])

    rows = []
    for key, sub in df.groupby(keys):
        wide = sub.pivot_table(index="example_id", columns="method", values="score")
        if args.baseline not in wide.columns:
            print(f"[warn] no baseline for {key}, skipped", file=sys.stderr)
            continue
        for method in wide.columns:
            pair = wide[[args.baseline, method]].dropna()
            if len(pair) < len(wide):
                print(f"[warn] {key} {method}: {len(wide) - len(pair)} unpaired examples dropped",
                      file=sys.stderr)
            base, meth = pair[args.baseline].to_numpy(), pair[method].to_numpy()
            mean, ci, delta, dci = paired_bootstrap(base, meth, args.n_boot, rng, args.alpha)
            rec = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
            rec.update(
                method=method, n=len(pair), mean=mean, ci_lo=ci[0], ci_hi=ci[1],
                delta=delta if method != args.baseline else 0.0,
                delta_lo=dci[0] if method != args.baseline else 0.0,
                delta_hi=dci[1] if method != args.baseline else 0.0,
            )
            rec["significant"] = (method != args.baseline) and not (rec["delta_lo"] <= 0 <= rec["delta_hi"])
            rows.append(rec)

    res = pd.DataFrame(rows)
    res.to_csv(args.out, index=False)

    # LaTeX-ready: mean with half-width of the 95% CI; dagger marks significant deltas.
    res["cell"] = res.apply(
        lambda r: f"{r['mean']:.2f}\\,{{\\scriptsize$\\pm${(r['ci_hi'] - r['ci_lo']) / 2:.2f}}}"
                  + ("$^\\dagger$" if r["significant"] else ""), axis=1)
    latex = res.pivot_table(index=["model", "method"], columns="dataset", values="cell",
                            aggfunc="first")
    latex.to_csv(args.out.replace(".csv", "_latex_cells.csv"))

    sig = res[(res["method"] != args.baseline)]
    print(f"\n{len(sig)} method/dataset pairs, {int(sig['significant'].sum())} with delta CI excluding 0")
    print(sig.sort_values("delta")[keys + ["method", "n", "delta", "delta_lo", "delta_hi", "significant"]]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
