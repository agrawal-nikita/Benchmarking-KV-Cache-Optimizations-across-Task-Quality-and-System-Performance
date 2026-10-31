"""
LongBench accuracy evaluation for kvpress (SnapKV / CaM).

Applies a kvpress press via `with press(model): model.generate(...)` so the
same prompt-building and scoring logic used by the KIVI and TurboQuant eval
scripts is used here, giving comparable numbers across methods.

Also supports LongBench-v2 (--dataset_names longbench_v2) once cached locally
via `python datasets/prepare_longbench_v2.py` (see that script's docstring);
pass --dataset_path pointing at the LongBench-v2 cache dir in that case.

Output layout (under benchmarking_accuracy/):
  {output_dir}/{press_name}/{model_tag}_cr{compression_ratio}/
    {dataset}.jsonl   — one JSON line per example: pred / answers / length
    results.json      — per-dataset scores + overall average
"""

import argparse
import json
import os
import random
import re
import string
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
KIVI_CONFIG_DIR = REPO_ROOT / "KIVI" / "config"

ALL_DATASETS = (
    "narrativeqa", "qasper", "multifieldqa_en",
    "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news",
    "trec", "triviaqa", "samsum",
    "lcc", "repobench-p",
)

METRIC_BY_DATASET = {
    "narrativeqa": "f1",
    "qasper": "f1",
    "multifieldqa_en": "f1",
    "hotpotqa": "f1",
    "2wikimqa": "f1",
    "musique": "f1",
    "gov_report": "rouge_l",
    "qmsum": "rouge_l",
    "multi_news": "rouge_l",
    "trec": "accuracy",
    "triviaqa": "f1",
    "samsum": "rouge_l",
    "lcc": "prefix_match",
    "repobench-p": "prefix_match",
    "longbench_v2": "mc_accuracy",
}


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def compute_f1(pred: str, ref: str) -> float:
    p_toks = normalize_text(pred).split()
    r_toks = normalize_text(ref).split()
    if not p_toks or not r_toks:
        return float(normalize_text(pred) == normalize_text(ref))
    common = Counter(p_toks) & Counter(r_toks)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec = n / len(p_toks)
    rec = n / len(r_toks)
    return 2 * prec * rec / (prec + rec)


def _lcs(a, b):
    m, n = len(a), len(b)
    if not m or not n:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            curr[j] = prev[j - 1] + 1 if a[i - 1] == b[j - 1] else max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]


def compute_rouge_l(pred: str, ref: str) -> float:
    p = normalize_text(pred).split()
    r = normalize_text(ref).split()
    if not p or not r:
        return 0.0
    lcs = _lcs(p, r)
    if lcs == 0:
        return 0.0
    prec = lcs / len(p)
    rec = lcs / len(r)
    return 2 * prec * rec / (prec + rec)


def compute_accuracy(pred: str, ref: str) -> float:
    return float(normalize_text(pred) == normalize_text(ref))


def compute_prefix_match(pred: str, ref: str) -> float:
    pred_lines = pred.strip().splitlines()
    ref_lines = ref.strip().splitlines()
    if not ref_lines:
        return 1.0 if not pred_lines else 0.0
    count = 0
    for p, r in zip(pred_lines, ref_lines):
        if p.strip() == r.strip():
            count += 1
        else:
            break
    return count / len(ref_lines)


def compute_mc_accuracy(pred: str, ref: str) -> float:
    # LongBench-v2 multiple choice: pred contains "The correct answer is (X)."
    # https://github.com/THUDM/LongBench/blob/main/pred.py (extract_answer)
    pred = pred.replace("*", "")
    if f"The correct answer is ({ref})" in pred or f"The correct answer is {ref}" in pred:
        return 1.0
    return 0.0


METRIC_FN = {
    "f1": compute_f1,
    "rouge_l": compute_rouge_l,
    "accuracy": compute_accuracy,
    "prefix_match": compute_prefix_match,
    "mc_accuracy": compute_mc_accuracy,
}


def score_prediction(dataset: str, pred: str, answers: list) -> float:
    fn = METRIC_FN[METRIC_BY_DATASET[dataset]]
    return max(fn(pred, ref) for ref in answers) if answers else 0.0


def build_chat(tokenizer, prompt: str, model_name: str, dataset: str, think: bool = False) -> str:
    if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
        return prompt
    model_lower = model_name.lower()
    if "qwen" in model_lower:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=think
        )
    if "instruct" in model_lower or "chat" in model_lower:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def load_longbench_dataset(dataset_repo_or_path, dataset_name: str):
    if dataset_repo_or_path and os.path.exists(dataset_repo_or_path):
        base_path = dataset_repo_or_path
        if os.path.isfile(base_path) and (base_path.endswith(".py") or base_path.endswith(".zip")):
            base_path = os.path.dirname(base_path)
        data_dir = os.path.join(base_path, "data")
        local_jsonl = os.path.join(data_dir, f"{dataset_name}.jsonl")
        if os.path.exists(local_jsonl):
            return load_dataset("json", data_files=local_jsonl, split="train")
        local_zip = os.path.join(base_path, "data.zip")
        if os.path.exists(local_zip):
            extract_dir = os.path.join(base_path, ".extracted_longbench", "data")
            extracted_jsonl = os.path.join(extract_dir, f"{dataset_name}.jsonl")
            if not os.path.exists(extracted_jsonl):
                os.makedirs(extract_dir, exist_ok=True)
                with zipfile.ZipFile(local_zip) as zf:
                    zf.extract(
                        f"data/{dataset_name}.jsonl",
                        path=os.path.join(base_path, ".extracted_longbench"),
                    )
            return load_dataset("json", data_files=extracted_jsonl, split="train")
    if dataset_name == "longbench_v2":
        # No local cache found; fall back to the HF hub directly.
        # Prefer `python datasets/prepare_longbench_v2.py` to cache it locally first
        # (this fallback re-downloads ~450MB on every call).
        ds = load_dataset("zai-org/LongBench-v2", split="train")
        return ds.map(lambda row: {"answers": [row["answer"]]})
    return load_dataset(dataset_repo_or_path or "zai-org/LongBench", dataset_name, split="test")


def get_model_input_device(model) -> torch.device:
    if hasattr(model, "hf_device_map"):
        embed_device = model.hf_device_map.get("model.embed_tokens")
        if embed_device is not None:
            return torch.device(embed_device)
    return next(model.parameters()).device


def maybe_truncate(tokenizer, prompt: str, max_tokens: int) -> str:
    ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(ids) <= max_tokens:
        return prompt
    half = max_tokens // 2
    return (
        tokenizer.decode(ids[:half], skip_special_tokens=True)
        + tokenizer.decode(ids[-half:], skip_special_tokens=True)
    )


@torch.no_grad()
def generate_one(model, tokenizer, press, prompt: str, max_new_tokens: int, max_ctx: int) -> str:
    device = get_model_input_device(model)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"].to(device)
    if input_ids.shape[1] > max_ctx - max_new_tokens:
        input_ids = input_ids[:, -(max_ctx - max_new_tokens):]
    attention_mask = torch.ones_like(input_ids)
    with press(model):
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
        )
    generated = outputs[0][input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def make_press(args):
    if args.press_name == "snapkv":
        from kvpress import SnapKVPress
        return SnapKVPress(
            compression_ratio=args.compression_ratio,
            window_size=args.window_size,
            kernel_size=args.kernel_size,
        )
    if args.press_name == "cam":
        from kvpress import CAMPress, SnapKVPress
        base_press = SnapKVPress(
            compression_ratio=args.compression_ratio,
            window_size=args.window_size,
            kernel_size=args.kernel_size,
        )
        return CAMPress(
            base_press=base_press,
            target_size=args.cam_target_size,
            compression_interval=args.cam_compression_interval,
            merge_budget=args.cam_merge_budget,
        )
    raise ValueError(f"Unknown press: {args.press_name!r}. Choose 'snapkv' or 'cam'.")


def parse_args():
    parser = argparse.ArgumentParser(description="kvpress LongBench accuracy evaluation")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--press_name", required=True, choices=["snapkv", "cam"])

    # SnapKV params (compression_ratio also used as CaM's base_press ratio)
    parser.add_argument("--compression_ratio", type=float, default=0.3)
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--kernel_size", type=int, default=5)

    # CaM-specific params
    parser.add_argument("--cam_target_size", type=int, default=2048)
    parser.add_argument("--cam_compression_interval", type=int, default=512)
    parser.add_argument("--cam_merge_budget", type=int, default=32)

    parser.add_argument("--dataset_path", default=str(REPO_ROOT / "datasets" / "LongBench"))
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Base output dir (default: <repo_root>/benchmarking_accuracy/{press_name})",
    )
    parser.add_argument("--dataset_names", nargs="+", default=list(ALL_DATASETS))
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    parser.add_argument(
        "--think", action="store_true",
        help="Enable Qwen3 thinking mode (enable_thinking=True in the chat template). No effect on non-Qwen models.",
    )
    parser.add_argument(
        "--think_max_new_tokens", type=int, default=4096,
        help="max_new_tokens used for every dataset when --think is set (overrides dataset2maxlen.json, "
             "since the reasoning trace itself can run into the thousands of tokens).",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="If set, only evaluate the first N examples of each dataset (for quick smoke tests).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(42)

    model_tag = args.model_name.rstrip("/").split("/")[-1]
    cr_str = f"cr{args.compression_ratio:.2f}".replace(".", "p")
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.press_name == "cam":
        config_str = f"{model_tag}_{cr_str}_ts{args.cam_target_size}_ci{args.cam_compression_interval}_mb{args.cam_merge_budget}"
    elif args.press_name == "snapkv":
        config_str = f"{model_tag}_{cr_str}_ws{args.window_size}_ks{args.kernel_size}"
    else:
        config_str = f"{model_tag}_{cr_str}"
    if args.think:
        config_str += "_think"
    if args.max_samples is not None:
        config_str += f"_n{args.max_samples}"
    config_str += f"_{ts}"

    if args.think and "qwen" not in args.model_name.lower():
        print(f"WARNING: --think has no effect on {args.model_name!r} (only Qwen chat templates support enable_thinking).")

    base_out = (
        args.output_dir
        if args.output_dir
        else str(REPO_ROOT / "benchmarking_accuracy" / args.press_name)
    )
    output_dir = Path(base_out) / config_str
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(KIVI_CONFIG_DIR / "dataset2prompt.json", "r") as f:
        dataset2prompt = json.load(f)
    with open(KIVI_CONFIG_DIR / "dataset2maxlen.json", "r") as f:
        dataset2maxlen = json.load(f)
    with open(KIVI_CONFIG_DIR / "model2maxlen.json", "r") as f:
        model2maxlen = json.load(f)

    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    model_kwargs = dict(
        pretrained_model_name_or_path=args.model_name,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    try:
        import flash_attn  # noqa: F401
        model_kwargs["attn_implementation"] = "flash_attention_2"
    except ImportError:
        pass
    if torch.cuda.device_count() > 1:
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = "cuda:0"

    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_ctx = model2maxlen.get(model_tag, 31500)
    press = make_press(args)
    if args.press_name == "cam":
        print(f"  Press:      cam  base_cr={args.compression_ratio}  target_size={args.cam_target_size}  compression_interval={args.cam_compression_interval}  merge_budget={args.cam_merge_budget}")
    else:
        print(f"  Press:      snapkv  compression_ratio={args.compression_ratio}  window_size={args.window_size}  kernel_size={args.kernel_size}")
    print(f"  max_ctx:    {max_ctx}")
    print(f"  Output:     {output_dir}\n")

    dataset_repo_or_path = args.dataset_path or "zai-org/LongBench"
    all_scores = {}
    all_breakdowns = {}

    for dataset in args.dataset_names:
        ds = load_longbench_dataset(dataset_repo_or_path, dataset)
        if args.max_samples is not None:
            ds = ds.select(range(min(args.max_samples, len(ds))))
        prompt_format = dataset2prompt[dataset]
        max_gen = args.think_max_new_tokens if args.think else dataset2maxlen[dataset]
        preds = []

        for row in tqdm(ds, desc=dataset, leave=False):
            prompt = prompt_format.format(**row)
            prompt = maybe_truncate(tokenizer, prompt, max_ctx - max_gen)
            prompt = build_chat(tokenizer, prompt, args.model_name, dataset, think=args.think)
            pred = generate_one(model, tokenizer, press, prompt, max_gen, max_ctx)
            answers = row.get("answers", [])
            if isinstance(answers, str):
                answers = [answers]
            entry = {
                "pred": pred,
                "answers": answers,
                "length": row.get("length", 0),
            }
            if dataset == "longbench_v2":
                entry["difficulty"] = row.get("difficulty")
            preds.append(entry)

        out_path = output_dir / f"{dataset}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for p in preds:
                json.dump(p, f, ensure_ascii=False)
                f.write("\n")

        scores = [score_prediction(dataset, p["pred"], p["answers"]) for p in preds]
        avg = sum(scores) / len(scores) if scores else 0.0
        all_scores[dataset] = round(avg * 100, 2)
        print(f"  {dataset:<30} {avg * 100:.1f}  ({METRIC_BY_DATASET[dataset]})")

        if dataset == "longbench_v2":
            breakdown = {}
            for group_key in ("difficulty", "length"):
                buckets = {}
                for p, s in zip(preds, scores):
                    buckets.setdefault(str(p.get(group_key)), []).append(s)
                breakdown[group_key] = {k: round(100 * sum(v) / len(v), 2) for k, v in buckets.items()}
            all_breakdowns[dataset] = breakdown

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    overall = sum(all_scores.values()) / len(all_scores) if all_scores else 0.0
    results_payload = {
        "model": args.model_name,
        "press": args.press_name,
        "compression_ratio": args.compression_ratio,
        **({"window_size": args.window_size, "kernel_size": args.kernel_size} if args.press_name == "snapkv" else {}),
        **({"cam_target_size": args.cam_target_size, "cam_compression_interval": args.cam_compression_interval, "cam_merge_budget": args.cam_merge_budget} if args.press_name == "cam" else {}),
        "think": args.think,
        "max_samples": args.max_samples,
        "scores": all_scores,
        "overall": round(overall, 2),
        **({"breakdowns": all_breakdowns} if all_breakdowns else {}),
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2, ensure_ascii=False)

    print(f"\nOverall: {overall:.1f}")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
