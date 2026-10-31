#!/usr/bin/env python3
"""
Export per-question quality scores for the confidence intervals (S4).
Self-contained: needs only  pip install pandas rouge fuzzywuzzy  (no kvpress).

Layouts read
------------
Study A:  results_studyA/<ID>/<model run folder>/<dataset>.jsonl
Study B:  results_studyB/<mode>/<ID>/<run folder>/samples.jsonl
          results_studyB/<mode>/<ID>/combined_samples*.jsonl
<ID> is one of FK Q1 Q2 Q3 Q4 E1 E2 M1 M2 (add other folder names to ALIASES).
A prompt-free copy "<name>.slim.jsonl" is used instead of "<name>.jsonl" if present.

Per line, the script uses
  - "score" if the harness stored one (Study B always; Study A if present),
  - otherwise it re-scores "pred" against "answers" with
      --scorer harness   (default) the metrics of benchmarking_eval_longbench.py, i.e. Table 3:
                         F1 (QA, TriviaQA), ROUGE-L (summarization, SAMSum), prefix match (code)
      --scorer longbench the official LongBench scorer (first-line truncation, rouge pkg, edit sim).
Questions are paired across runs by "index" (or "_id").

Usage
-----
  python export_scores.py --results results_studyA --out scores_studyA.csv
  python export_scores.py --results results_studyB/reasoning_on --out scores_B_on.csv
"""
import argparse
import json
import re
import string
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

IDS = {"FK", "Q1", "Q2", "Q3", "Q4", "E1", "E2", "M1", "M2"}
ALIASES = {"M1_rerun": "M1"}           # other folder name -> ID

# ---------------- LongBench metrics (English datasets) ----------------
def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(pred_tokens, gt_tokens):
    common = Counter(pred_tokens) & Counter(gt_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    p, r = same / len(pred_tokens), same / len(gt_tokens)
    return 2 * p * r / (p + r)


def qa_f1_score(pred, gt, **kw):
    return f1_score(normalize_answer(pred).split(), normalize_answer(gt).split())


def rouge_score(pred, gt, **kw):
    from rouge import Rouge
    try:
        return Rouge().get_scores([pred], [gt], avg=True)["rouge-l"]["f"]
    except Exception:
        return 0.0


def code_sim_score(pred, gt, **kw):
    from fuzzywuzzy import fuzz
    line = ""
    for l in pred.lstrip("\n").split("\n"):
        if "`" not in l and "#" not in l and "//" not in l:
            line = l
            break
    return fuzz.ratio(line, gt) / 100


def classification_score(pred, gt, **kw):
    matches = [c for c in kw.get("all_classes") or [] if c in pred]
    matches = [m for m in matches if not (m in gt and m != gt)]
    return 1.0 / len(matches) if gt in matches else 0.0


METRIC = {
    "narrativeqa": qa_f1_score, "qasper": qa_f1_score, "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score, "2wikimqa": qa_f1_score, "musique": qa_f1_score,
    "triviaqa": qa_f1_score,
    "gov_report": rouge_score, "qmsum": rouge_score, "multi_news": rouge_score, "samsum": rouge_score,
    "lcc": code_sim_score, "repobench-p": code_sim_score,
}
FIRST_LINE = {"trec", "triviaqa", "samsum", "lsht"}
STUDY_A_DATASETS = set(METRIC)           # the 13 datasets of the paper (trec excluded)


# ---------------- Harness metrics (benchmarking_eval_longbench.py, used for Table 3) ----------------
def h_normalize(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def h_f1(pred, ref):
    p, r = h_normalize(pred).split(), h_normalize(ref).split()
    if not p or not r:
        return float(h_normalize(pred) == h_normalize(ref))
    n = sum((Counter(p) & Counter(r)).values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(r)
    return 2 * prec * rec / (prec + rec)


def _h_lcs(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        curr = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            curr[j] = prev[j - 1] + 1 if a[i - 1] == b[j - 1] else max(curr[j - 1], prev[j])
        prev = curr
    return prev[-1]


def h_rouge_l(pred, ref):
    p, r = h_normalize(pred).split(), h_normalize(ref).split()
    if not p or not r:
        return 0.0
    l = _h_lcs(p, r)
    if l == 0:
        return 0.0
    prec, rec = l / len(p), l / len(r)
    return 2 * prec * rec / (prec + rec)


def h_prefix_match(pred, ref):
    pl, rl = pred.strip().splitlines(), ref.strip().splitlines()
    if not rl:
        return 1.0 if not pl else 0.0
    c = 0
    for p, r in zip(pl, rl):
        if p.strip() == r.strip():
            c += 1
        else:
            break
    return c / len(rl)


H_METRIC = {
    "narrativeqa": h_f1, "qasper": h_f1, "multifieldqa_en": h_f1, "hotpotqa": h_f1,
    "2wikimqa": h_f1, "musique": h_f1, "triviaqa": h_f1,
    "gov_report": h_rouge_l, "qmsum": h_rouge_l, "multi_news": h_rouge_l, "samsum": h_rouge_l,
    "lcc": h_prefix_match, "repobench-p": h_prefix_match,
}


def harness_score(task, pred, answers):
    return max((H_METRIC[task](pred, ref) for ref in answers), default=0.0)


SCORER = "harness"     # set from --scorer


def longbench_score(task, pred, answers, all_classes):
    if task in FIRST_LINE:
        pred = pred.lstrip("\n").split("\n")[0]
    return max((METRIC[task](pred, gt, all_classes=all_classes) for gt in answers), default=0.0)


def v2_correct(pred, gold):
    p = pred.replace("*", "")
    return float(f"The correct answer is ({gold})" in p or f"The correct answer is {gold}" in p)
# ----------------------------------------------------------------------


def config_id(path: Path):
    for part in path.parts:
        if part in IDS:
            return part
        if part in ALIASES:
            return ALIASES[part]
    return None


def model_of(path: Path):
    for part in path.parts:
        p = part.lower()
        if "llama" in p:
            return "llama"
        if "30b" in p:
            return "qwen30b"
        if "qwen" in p:
            return "qwen"
    return "unknown"


def as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def prefer_slim(files):
    files = sorted(set(files))
    return [f for f in files if f.name.endswith(".slim.jsonl")
            or not f.with_name(f.name.replace(".jsonl", ".slim.jsonl")).exists()]


def read_jsonl(path):
    bad = 0
    with open(path) as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            try:
                yield i, json.loads(line)
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"[warn] {path}: {bad} unreadable line(s) (incomplete file?)", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--out", default="scores.csv")
    ap.add_argument("--kvpress", help="ignored (kept for compatibility)")
    ap.add_argument("--scorer", choices=["harness", "longbench"], default="harness",
                    help="harness = metrics of benchmarking_eval_longbench.py (Table 3); "
                         "longbench = official LongBench scorer")
    args = ap.parse_args()
    global SCORER
    SCORER = args.scorer
    print(f"[info] scorer: {SCORER}", file=sys.stderr)

    rows, skipped = [], set()
    files = prefer_slim(args.results.rglob("*.jsonl"))
    for jf in files:
        cid = config_id(jf.relative_to(args.results))
        if cid is None:
            skipped.add(jf.parent); continue
        stem = jf.name.removesuffix(".slim.jsonl").removesuffix(".jsonl")
        is_v2 = stem.startswith("samples") or stem.startswith("combined_samples")
        task = stem.removesuffix("-e")
        if not is_v2 and task not in STUDY_A_DATASETS:
            continue                                            # e.g. trec
        n = 0
        for i, r in read_jsonl(jf):
            n += 1
            ex = r.get("index", r.get("_id", r.get("id", i)))
            pred = str(r.get("pred", r.get("predicted_answer", r.get("prediction", ""))))
            if is_v2:
                gold = as_list(r.get("answers", r.get("answer")))
                gold = gold[0] if gold else None
                s = float(r["score"]) if "score" in r else v2_correct(pred, gold)
                ct = r.get("context_tokens")
                grp = r.get("length") or ("short" if ct is not None and ct < 32_000 else
                                          "medium" if ct is not None and ct < 128_000 else
                                          "long" if ct is not None else "")
                rows.append(dict(model=model_of(jf.relative_to(args.results)), method=cid,
                                 dataset="longbench-v2", group=grp, example_id=ex, score=s,
                                 no_answer=float("The correct answer is" not in pred.replace("*", "")),
                                 ref=str(gold)))
            else:
                answers = as_list(r.get("answers", r.get("answer")))
                if SCORER == "harness" and "score" in r and r["score"] is not None:
                    s = float(r["score"])
                    s = s / 100 if s > 1.0 else s                  # harness may store 0-100
                elif "pred" in r or "predicted_answer" in r or "prediction" in r:
                    s = (harness_score(task, pred, answers) if SCORER == "harness"
                         else longbench_score(task, pred, answers, as_list(r.get("all_classes"))))
                else:
                    print(f"[warn] {jf}: line {i} has neither 'score' nor 'pred'; file skipped "
                          f"(system-metrics log?)", file=sys.stderr)
                    break
                rows.append(dict(model=model_of(jf.relative_to(args.results)), method=cid,
                                 dataset=task, example_id=ex, score=s, ref=str(answers)[:80]))
        print(f"[read] {cid:3s} {jf.relative_to(args.results)}: {n} lines", file=sys.stderr)

    if not rows:
        sys.exit("no scores found: check --results and the folder names (FK, Q1, ...)")
    out = pd.DataFrame(rows)
    bad = (out.groupby(["model", "dataset", "example_id"])["ref"].nunique() > 1).sum()
    if bad:
        print(f"[warn] {bad} question ids have different reference answers across runs: "
              f"pairing by index is not valid for those datasets", file=sys.stderr)
    out.drop(columns="ref").to_csv(args.out, index=False)
    print(f"\nwrote {len(out)} rows to {args.out}\n")
    print("Mean score x100 per configuration and dataset (compare with Table 3):")
    print(out.groupby(["model", "dataset", "method"])["score"].mean().mul(100).round(2)
          .unstack("method").to_string())
    print("\nQuestions per configuration and dataset (should be equal within a row):")
    print(out.groupby(["model", "dataset", "method"]).size().unstack("method").to_string())
    if skipped:
        print(f"\n[info] skipped folders without an ID: {sorted(map(str, skipped))[:5]}", file=sys.stderr)


if __name__ == "__main__":
    main()
