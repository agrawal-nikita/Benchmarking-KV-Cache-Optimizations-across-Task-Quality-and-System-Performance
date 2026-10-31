import argparse
import json
import os
import re
import string
import zipfile
from collections import Counter

import numpy as np
import random
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
)
from transformers.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

os.environ["WANDB_DISABLED"] = "true"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(SCRIPT_DIR, "config")

ALL_DATASETS = (
    "narrativeqa", "qasper", "multifieldqa_en",
    "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news", 
    "triviaqa", "samsum",
    "lcc", "repobench-p"
)

METRIC_BY_DATASET = {
    "narrativeqa": "f1",
    "qasper": "f1",
    "multifieldqa_en": "f1",
    "multifieldqa_zh": "f1",
    "hotpotqa": "f1",
    "2wikimqa": "f1",
    "musique": "f1",
    "dureader": "rouge_l",
    "gov_report": "rouge_l",
    "qmsum": "rouge_l",
    "multi_news": "rouge_l",
    "vcsum": "rouge_l",
    "trec": "accuracy",
    "triviaqa": "f1",
    "samsum": "rouge_l",
    "lsht": "accuracy",
    "passage_count": "accuracy",
    "passage_retrieval_en": "accuracy",
    "passage_retrieval_zh": "accuracy",
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


def resolve_model_path(model_name_or_path: str) -> str:
    import glob
    model2path_path = os.path.join(CONFIG_DIR, "model2path.json")
    if os.path.exists(model2path_path):
        with open(model2path_path, "r") as f:
            model2path = json.load(f)
        resolved = model2path.get(model_name_or_path, model_name_or_path)
    else:
        resolved = model_name_or_path
    if os.path.isdir(resolved):
        return resolved
    if "/" in resolved and not os.path.exists(resolved):
        org, repo = resolved.split("/", 1)
        pattern = os.path.expanduser(
            f"~/.cache/huggingface/hub/models--{org}--{repo}/snapshots/*"
        )
        snapshots = sorted(
            p for p in glob.glob(pattern)
            if os.path.exists(os.path.join(p, "config.json"))
        )
        if snapshots:
            return snapshots[-1]
    return resolved


def build_chat(tokenizer, prompt: str, model_name: str, dataset: str, think: bool = False) -> str:
    if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
        return prompt
    model_lower = model_name.lower()
    if "longchat" in model_lower:
        try:
            from fastchat.model import get_conversation_template
            conv = get_conversation_template("vicuna")
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            return conv.get_prompt()
        except ImportError:
            pass
    if "qwen" in model_lower:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=think
        )
    if "instruct" in model_lower or "chat" in model_lower:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def post_process(response: str, model_name: str) -> str:
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


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


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def parse_args():
    parser = argparse.ArgumentParser(description="KIVI LongBench accuracy evaluation")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--k_bits", type=int, default=2)
    parser.add_argument("--v_bits", type=int, default=2)
    parser.add_argument("--group_size", type=int, default=32)
    parser.add_argument("--residual_length", type=int, default=32)
    parser.add_argument("--use_flash", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Base output dir (default: <repo_root>/benchmarking_accuracy/kivi)",
    )
    parser.add_argument("--dataset_names", nargs="+", default=list(ALL_DATASETS))
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
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


def build_model_and_tokenizer(args, resolved_path: str):
    model_name_lower = args.model_name_or_path.lower()
    if "mistral" in model_name_lower:
        config = MistralConfig.from_pretrained(
            resolved_path,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        )
    elif "a3b" in model_name_lower:
        config = Qwen3MoeConfig.from_pretrained(
            resolved_path,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        )
    elif "qwen" in model_name_lower:
        config = Qwen3Config.from_pretrained(
            resolved_path,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        )
    else:
        config = LlamaConfig.from_pretrained(
            resolved_path,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        )

    model_kwargs = dict(
        pretrained_model_name_or_path=resolved_path,
        config=config,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    if torch.cuda.device_count() > 1:
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = "cuda:0"

    if args.k_bits in (2, 4) and args.v_bits in (2, 4):
        config.k_bits = args.k_bits
        config.v_bits = args.v_bits
        config.group_size = args.group_size
        config.residual_length = args.residual_length
        config.use_flash = args.use_flash
        config._flash_attn_2_enabled = args.use_flash
        if "mistral" in model_name_lower:
            from models.mistral_kivi import MistralForCausalLM_KIVI
            model = MistralForCausalLM_KIVI.from_pretrained(**model_kwargs)
        elif "a3b" in model_name_lower:
            from models.qwen_moe_kivi import Qwen3MoeForCausalLM_KIVI
            model = Qwen3MoeForCausalLM_KIVI.from_pretrained(**model_kwargs)
        elif "qwen" in model_name_lower:
            from models.qwen_kivi import Qwen3ForCausalLM_KIVI
            model = Qwen3ForCausalLM_KIVI.from_pretrained(**model_kwargs)
        else:
            if "llama-3.1" in model_name_lower:
                from models.llama_kivi_31 import LlamaForCausalLM_KIVI_31 as LlamaForCausalLM_KIVI
            else:
                from models.llama_kivi import LlamaForCausalLM_KIVI
            model = LlamaForCausalLM_KIVI.from_pretrained(**model_kwargs)
    elif args.k_bits >= 16 and args.v_bits >= 16:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        if "mistral" in model_name_lower:
            model = MistralForCausalLM.from_pretrained(**model_kwargs)
        elif "a3b" in model_name_lower:
            model = Qwen3MoeForCausalLM.from_pretrained(**model_kwargs)
        elif "qwen" in model_name_lower:
            model = Qwen3ForCausalLM.from_pretrained(**model_kwargs)
        else:
            model = LlamaForCausalLM.from_pretrained(**model_kwargs)
    else:
        raise ValueError(
            f"Unsupported KV bit-widths: k_bits={args.k_bits}, v_bits={args.v_bits}. "
            "Use 2 or 4 for KIVI quantization, or 16 for full-precision baseline."
        )

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_path,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_pred(model, tokenizer, data, max_length: int, max_gen: int,
             prompt_format: str, dataset: str, device, model_name: str, think: bool = False) -> list:
    preds = []
    for json_obj in tqdm(data, desc=dataset, leave=False):
        prompt = prompt_format.format(**json_obj)
        tokenized = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized) > max_length:
            half = max_length // 2
            prompt = (
                tokenizer.decode(tokenized[:half], skip_special_tokens=True)
                + tokenizer.decode(tokenized[-half:], skip_special_tokens=True)
            )
        prompt = build_chat(tokenizer, prompt, model_name, dataset, think=think)
        inputs = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = inputs.input_ids.shape[-1]
        gen_kwargs = dict(
            max_new_tokens=max_gen,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
        )
        if dataset == "samsum":
            gen_kwargs["min_length"] = context_length + 1
            gen_kwargs["eos_token_id"] = [
                tokenizer.eos_token_id,
                tokenizer.encode("\n", add_special_tokens=False)[-1],
            ]
        output = model.generate(**inputs, **gen_kwargs)[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        answers = json_obj.get("answers", [])
        if isinstance(answers, str):
            answers = [answers]
        entry = {
            "pred": pred,
            "answers": answers,
            "length": json_obj.get("length", 0),
        }
        if dataset == "longbench_v2":
            entry["difficulty"] = json_obj.get("difficulty")
        preds.append(entry)
    return preds


def main():
    args = parse_args()
    seed_everything(args.seed)

    resolved_path = resolve_model_path(args.model_name_or_path)
    model_name = args.model_name_or_path
    tag = model_name.rstrip("/").split("/")[-1]

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.k_bits < 16:
        config_str = f"{tag}_{args.k_bits}bits_kv_group{args.group_size}_residual{args.residual_length}"
    else:
        config_str = f"{tag}_fp16"
    if args.think:
        config_str += "_think"
    if args.max_samples is not None:
        config_str += f"_n{args.max_samples}"
    config_str += f"_{ts}"

    if args.think and "qwen" not in model_name.lower():
        print(f"WARNING: --think has no effect on {model_name!r} (only Qwen chat templates support enable_thinking).")

    repo_root = os.path.dirname(SCRIPT_DIR)
    base_out = args.output_dir if args.output_dir else os.path.join(repo_root, "benchmarking_accuracy", "kivi")
    output_dir = os.path.join(base_out, config_str)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(CONFIG_DIR, "model2maxlen.json"), "r") as f:
        model2maxlen = json.load(f)
    with open(os.path.join(CONFIG_DIR, "dataset2prompt.json"), "r") as f:
        dataset2prompt = json.load(f)
    with open(os.path.join(CONFIG_DIR, "dataset2maxlen.json"), "r") as f:
        dataset2maxlen = json.load(f)

    max_length = model2maxlen.get(tag, 31500)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model: {model_name}")
    model, tokenizer = build_model_and_tokenizer(args, resolved_path)
    print(f"  max_context={max_length}, k_bits={args.k_bits}, v_bits={args.v_bits}")
    print(f"  Output: {output_dir}\n")

    dataset_repo_or_path = args.dataset_path or "zai-org/LongBench"
    all_scores = {}
    all_breakdowns = {}

    for dataset in args.dataset_names:
        data = load_longbench_dataset(dataset_repo_or_path, dataset)
        if args.max_samples is not None:
            data = data.select(range(min(args.max_samples, len(data))))
        prompt_format = dataset2prompt[dataset]
        max_gen = args.think_max_new_tokens if args.think else dataset2maxlen[dataset]
        preds = get_pred(
            model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name,
            think=args.think,
        )

        out_path = os.path.join(output_dir, f"{dataset}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
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
        "model": model_name,
        "k_bits": args.k_bits,
        "v_bits": args.v_bits,
        "group_size": args.group_size,
        "residual_length": args.residual_length,
        "think": args.think,
        "max_samples": args.max_samples,
        "scores": all_scores,
        "overall": round(overall, 2),
        **({"breakdowns": all_breakdowns} if all_breakdowns else {}),
    }
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2, ensure_ascii=False)

    print(f"\nOverall: {overall:.1f}")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
