"""
Download LongBench-v2 once and cache it locally in the same
data/{dataset_name}.jsonl layout that KIVI/pred_long_bench.py and
kvpress/benchmarking_eval_longbench.py already know how to read via
load_longbench_dataset() (no code changes needed on the loader side).

Usage:
    python datasets/prepare_longbench_v2.py

Then evaluate with, e.g.:
    ./run_benchmark.sh eval-kivi \
        --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
        --dataset_path datasets/LongBench-v2 \
        --dataset_names longbench_v2 \
        --k_bits 2 --v_bits 2

    ./run_benchmark.sh eval-snapkv \
        --model_name meta-llama/Llama-3.1-8B-Instruct \
        --dataset_path datasets/LongBench-v2 \
        --dataset_names longbench_v2 \
        --compression_ratio 0.75
"""

import json
import os

from datasets import load_dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "LongBench-v2", "data")
OUT_PATH = os.path.join(OUT_DIR, "longbench_v2.jsonl")

# zai-org/LongBench-v2 is the renamed THUDM/LongBench-v2 repo.
HF_REPO = "zai-org/LongBench-v2"

FIELDS = (
    "_id", "domain", "sub_domain", "difficulty", "length",
    "question", "choice_A", "choice_B", "choice_C", "choice_D",
    "answer", "context",
)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Downloading {HF_REPO} (one-time, ~450MB)...")
    ds = load_dataset(HF_REPO, split="train")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in ds:
            record = {k: row[k] for k in FIELDS}
            # Mirror LongBench-v1's "answers" list so the existing eval scripts'
            # `answers = json_obj.get("answers", [])` path works unmodified.
            record["answers"] = [row["answer"]]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(ds)} examples to {OUT_PATH}")


if __name__ == "__main__":
    main()
