"""
LongBench-v2 performance + accuracy benchmarking for kvpress SnapKV.

Dedicated to LongBench-v2 (unlike benchmarking_snapkv_kvpress.py, which covers
LongBench-v1's shorter tasks) so that context_length_buckets can reflect
LongBench-v2's actual range: 0-32k / 32k-128k / 128k+ tokens, instead of
v1's 0-4k / 4k-8k / 8k+.

Reports TTFT, decode throughput, prefill + decode KV cache footprint, and
accuracy (scored the same way as benchmarking_eval_longbench.py --press_name
snapkv) in one run. Supports --think for Qwen3 thinking-mode comparisons.

First cache the dataset locally: python datasets/prepare_longbench_v2.py
"""

import argparse
import glob
import json
import os
import platform
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from kvpress import SnapKVPress

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
KIVI_CONFIG_DIR = os.path.join(ROOT_DIR, "KIVI", "config")
DEFAULT_DATASET_PATH = str(Path(ROOT_DIR) / "datasets" / "LongBench-v2")
DATASET_NAME = "longbench_v2"


def is_cuda_oom(error: BaseException) -> bool:
    return "out of memory" in str(error).lower() and "cuda" in str(error).lower()


def cleanup_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def resolve_model_path(model_name_or_path: str) -> str:
    model2path_path = os.path.join(KIVI_CONFIG_DIR, "model2path.json")
    if os.path.exists(model2path_path):
        with open(model2path_path, "r", encoding="utf-8") as f:
            model2path = json.load(f)
        if model_name_or_path in model2path:
            return model2path[model_name_or_path]
        lower_model2path = {key.lower(): value for key, value in model2path.items()}
        return lower_model2path.get(model_name_or_path.lower(), model_name_or_path)
    return model_name_or_path


def resolve_local_snapshot_path(model_name_or_path: str, cache_dir: Optional[str]) -> str:
    if os.path.exists(model_name_or_path):
        return model_name_or_path
    if not cache_dir or "/" not in model_name_or_path:
        return model_name_or_path

    org, repo = model_name_or_path.split("/", 1)
    snapshot_glob = os.path.join(cache_dir, f"models--{org}--{repo}", "snapshots", "*")
    snapshots = [
        path
        for path in glob.glob(snapshot_glob)
        if os.path.exists(os.path.join(path, "config.json"))
    ]
    if not snapshots:
        return model_name_or_path
    snapshots.sort(key=os.path.getmtime, reverse=True)
    return snapshots[0]


def cuda_time_ms(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return out, float(start.elapsed_time(end))


def compute_last_token_logits(model, hidden_states: torch.Tensor) -> torch.Tensor:
    pretraining_tp = getattr(model.config, "pretraining_tp", 1)
    if pretraining_tp > 1:
        lm_head_slices = model.lm_head.weight.split(model.vocab_size // pretraining_tp, dim=0)
        logits = [torch.nn.functional.linear(hidden_states, lm_head_slices[i]) for i in range(pretraining_tp)]
        return torch.cat(logits, dim=-1).float()
    return model.lm_head(hidden_states).float()


@dataclass
class SampleMetrics:
    context_tokens: int
    ttft_ms: float
    output_throughput_toks_per_s: Optional[float]
    prefill_kv_memory_mb: float
    cache_size_mb: float  # KV cache size immediately after prefill
    compressed_cache_tokens: Optional[int]  # cache seq length immediately after prefill
    decode_cache_size_mb: float  # KV cache size at the end of decoding
    decode_compressed_cache_tokens: Optional[int]  # cache seq length at the end of decoding
    pred: str = ""


def compute_cache_size_mb(cache) -> float:
    seen_tensors = set()

    def tensor_bytes(tensor: torch.Tensor) -> int:
        tensor_id = (tensor.data_ptr(), tensor.numel(), tensor.element_size())
        if tensor_id in seen_tensors:
            return 0
        seen_tensors.add(tensor_id)
        return tensor.numel() * tensor.element_size()

    def walk(obj) -> int:
        if obj is None:
            return 0
        if torch.is_tensor(obj):
            return tensor_bytes(obj)
        if isinstance(obj, (list, tuple)):
            return sum(walk(item) for item in obj)
        if isinstance(obj, dict):
            return sum(walk(item) for item in obj.values())

        total = 0
        for attr in (
            "key_cache",
            "value_cache",
            "keys",
            "values",
            "_quantized_keys",
            "_quantized_values",
            "layers",
        ):
            if hasattr(obj, attr):
                total += walk(getattr(obj, attr))
        return total

    return walk(cache) / (1024**2)


def model_tag(model_name: str) -> str:
    return model_name.rstrip("/").split("/")[-1]


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def build_model_and_tokenizer(
    model_name: str,
    cache_dir: Optional[str],
    local_files_only: bool,
    dtype_str: str,
):
    model_name = resolve_model_path(model_name)
    if local_files_only:
        model_name = resolve_local_snapshot_path(model_name, cache_dir)
    model_kwargs = dict(
        pretrained_model_name_or_path=model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        torch_dtype=get_torch_dtype(dtype_str),
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    if torch.cuda.device_count() > 1:
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = "cuda:0"

    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_model_input_device(model) -> torch.device:
    if hasattr(model, "hf_device_map"):
        embed_device = model.hf_device_map.get("model.embed_tokens")
        if embed_device is not None:
            return torch.device(embed_device)
    return next(model.parameters()).device


def build_chat(tokenizer, prompt: str, model_name: str, think: bool = False) -> str:
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


def compute_mc_accuracy(pred: str, ref: str) -> float:
    # LongBench-v2 multiple choice: pred contains "The correct answer is (X)."
    pred = pred.replace("*", "")
    if f"The correct answer is ({ref})" in pred or f"The correct answer is {ref}" in pred:
        return 1.0
    return 0.0


def score_prediction(pred: str, answers: list) -> float:
    return max(compute_mc_accuracy(pred, ref) for ref in answers) if answers else 0.0


def load_longbench_v2_dataset(dataset_path: Optional[str]):
    dataset_path = dataset_path or DEFAULT_DATASET_PATH
    if os.path.exists(dataset_path):
        base_path = dataset_path
        if os.path.isfile(base_path):
            base_path = os.path.dirname(base_path)
        local_jsonl = os.path.join(base_path, "data", f"{DATASET_NAME}.jsonl")
        if os.path.exists(local_jsonl):
            return load_dataset("json", data_files=local_jsonl, split="train")
    # No local cache found; fall back to the HF hub directly.
    # Prefer `python datasets/prepare_longbench_v2.py` to cache it locally first
    # (this fallback re-downloads ~450MB on every call).
    ds = load_dataset("zai-org/LongBench-v2", split="train")
    return ds.map(lambda row: {"answers": [row["answer"]]})


def maybe_truncate_prompt(tokenizer, prompt: str, max_prompt_tokens: Optional[int]) -> str:
    if max_prompt_tokens is None:
        return prompt

    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(tokenized_prompt) <= max_prompt_tokens:
        return prompt

    half = max_prompt_tokens // 2
    return (
        tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)
        + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
    )


def load_prompts(
    dataset_path: Optional[str],
    tokenizer,
    model_name: str,
    max_prompt_tokens: Optional[int],
    think: bool = False,
) -> List[Dict[str, Any]]:
    dataset2prompt_path = os.path.join(KIVI_CONFIG_DIR, "dataset2prompt.json")
    with open(dataset2prompt_path, "r", encoding="utf-8") as f:
        dataset2prompt = json.load(f)
    prompt_format = dataset2prompt[DATASET_NAME]
    ds = load_longbench_v2_dataset(dataset_path)
    prompts = []
    for row in ds:
        context_tokens = len(tokenizer(row["context"], truncation=False, return_tensors="pt").input_ids[0])
        prompt = prompt_format.format(**row)
        prompt = maybe_truncate_prompt(tokenizer, prompt, max_prompt_tokens)
        prompt = build_chat(tokenizer, prompt, model_name, think=think)
        answers = row.get("answers", [])
        if isinstance(answers, str):
            answers = [answers]
        prompts.append({
            "prompt": prompt,
            "context_tokens": context_tokens,
            "answers": answers,
            "difficulty": row.get("difficulty"),
            "length_category": row.get("length"),
        })
    return prompts


@torch.inference_mode()
def benchmark_one(
    model,
    tokenizer,
    press: SnapKVPress,
    prompt: str,
    max_new_tokens: int,
    context_tokens: int,
) -> SampleMetrics:
    cleanup_cuda()
    inputs = tokenizer(prompt, return_tensors="pt")
    input_device = get_model_input_device(model)
    input_ids = inputs["input_ids"].to(input_device)
    prompt_length = input_ids.shape[1]

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated(input_device)
    cache = DynamicCache()

    def do_prefill():
        with press(model):
            return model.model(
                input_ids=input_ids,
                past_key_values=cache,
                return_dict=True,
                use_cache=True,
            )

    decoder_outputs, prefill_latency_ms = cuda_time_ms(do_prefill)
    last_hidden_state = decoder_outputs[0][:, -1:, :].contiguous()
    cache_size_mb = compute_cache_size_mb(cache)
    compressed_cache_tokens = cache.get_seq_length() if hasattr(cache, "get_seq_length") else None
    del decoder_outputs

    torch.cuda.synchronize()
    mem_after_prefill = torch.cuda.memory_allocated(input_device)
    prefill_kv_memory_mb = max(0, mem_after_prefill - mem_before) / (1024**2)

    def pick_first_token():
        logits = compute_last_token_logits(model, last_hidden_state)
        return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    first_token, _ = cuda_time_ms(pick_first_token)
    generated_token_ids: List[int] = [int(first_token.item())]

    generated_tokens = 1
    current_input_ids = first_token
    current_position_ids = torch.tensor([[prompt_length]], device=input_device, dtype=torch.long)
    decode_step_times_ms: List[float] = []
    eos_token_ids = tokenizer.eos_token_id
    if eos_token_ids is None:
        eos_token_ids = []
    elif not isinstance(eos_token_ids, list):
        eos_token_ids = [eos_token_ids]
    should_continue_decoding = int(first_token.item()) not in eos_token_ids

    if max_new_tokens > 1 and should_continue_decoding:
        def do_decode():
            return model(
                input_ids=current_input_ids,
                past_key_values=cache,
                position_ids=current_position_ids,
                use_cache=True,
                return_dict=True,
            )

        step_outputs, first_decode_ms = cuda_time_ms(do_decode)
        decode_step_times_ms.append(first_decode_ms)
        ttft_ms = prefill_latency_ms + first_decode_ms

        next_token = torch.argmax(step_outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_tokens += 1
        generated_token_ids.append(int(next_token.item()))
        should_continue_decoding = int(next_token.item()) not in eos_token_ids
        current_input_ids = next_token
        current_position_ids = current_position_ids + 1
    else:
        ttft_ms = prefill_latency_ms

    for _ in range(max(0, max_new_tokens - generated_tokens)):
        if not should_continue_decoding:
            break

        def do_decode():
            return model(
                input_ids=current_input_ids,
                past_key_values=cache,
                position_ids=current_position_ids,
                use_cache=True,
                return_dict=True,
            )

        step_outputs, step_ms = cuda_time_ms(do_decode)
        decode_step_times_ms.append(step_ms)

        next_token = torch.argmax(step_outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_tokens += 1
        generated_token_ids.append(int(next_token.item()))
        should_continue_decoding = int(next_token.item()) not in eos_token_ids
        current_input_ids = next_token
        current_position_ids = current_position_ids + 1

    decode_time_ms = sum(decode_step_times_ms)
    output_throughput = None
    if decode_time_ms > 0 and decode_step_times_ms:
        output_throughput = len(decode_step_times_ms) / (decode_time_ms / 1000.0)

    decode_cache_size_mb = compute_cache_size_mb(cache)
    decode_compressed_cache_tokens = cache.get_seq_length() if hasattr(cache, "get_seq_length") else None
    pred = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

    del last_hidden_state
    del cache
    torch.cuda.synchronize()

    return SampleMetrics(
        context_tokens=context_tokens,
        ttft_ms=ttft_ms,
        output_throughput_toks_per_s=output_throughput,
        prefill_kv_memory_mb=prefill_kv_memory_mb,
        cache_size_mb=cache_size_mb,
        compressed_cache_tokens=compressed_cache_tokens,
        decode_cache_size_mb=decode_cache_size_mb,
        decode_compressed_cache_tokens=decode_compressed_cache_tokens,
        pred=pred,
    )


def summarize_basic(results: List[SampleMetrics]) -> Dict[str, Any]:
    def values(name: str) -> List[float]:
        return [v for r in results if (v := getattr(r, name)) is not None]

    def mean_or_none(name: str) -> Optional[float]:
        vals = values(name)
        return float(statistics.mean(vals)) if vals else None

    return {
        "mean_ttft_ms": mean_or_none("ttft_ms"),
        "mean_output_throughput_toks_per_s": mean_or_none("output_throughput_toks_per_s"),
        "mean_prefill_kv_memory_mb": mean_or_none("prefill_kv_memory_mb"),
        "mean_cache_size_mb": mean_or_none("cache_size_mb"),
        "mean_compressed_cache_tokens": mean_or_none("compressed_cache_tokens"),
        "mean_decode_cache_size_mb": mean_or_none("decode_cache_size_mb"),
        "mean_decode_compressed_cache_tokens": mean_or_none("decode_compressed_cache_tokens"),
    }


def get_context_length_buckets() -> List[Tuple[str, int, Optional[int]]]:
    # LongBench-v2's actual context range, unlike v1's 0-4k/4k-8k/8k+.
    return [("0_32k", 0, 32000), ("32k_128k", 32000, 128000), ("128k_plus", 128000, None)]


def summarize(results: List[SampleMetrics]) -> Dict[str, Any]:
    summary = summarize_basic(results)
    bucket_summaries: Dict[str, Any] = {}
    for bucket_name, lower, upper in get_context_length_buckets():
        bucket_results = [
            r for r in results if r.context_tokens >= lower and (upper is None or r.context_tokens < upper)
        ]
        bucket_summary = summarize_basic(bucket_results)
        bucket_summary["min_context_tokens"] = min((r.context_tokens for r in bucket_results), default=None)
        bucket_summary["max_context_tokens"] = max((r.context_tokens for r in bucket_results), default=None)
        bucket_summaries[bucket_name] = bucket_summary
    summary["context_length_buckets"] = bucket_summaries
    return summary


def get_system_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "gpu_name": props.name,
                "gpu_total_memory_gb": round(props.total_memory / (1024**3), 2),
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_version": torch.version.cuda,
            }
        )
    return info


def build_output_paths(args) -> Dict[str, str]:
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    think_suffix = "_think" if args.think else ""
    run_dir = os.path.join(
        args.output_dir,
        (
            f"{model_tag(args.model_name)}_kvpress_snapkv_longbenchv2_ratio{args.compression_ratio}"
            f"_win{args.window_size}_kernel{args.kernel_size}_{args.dtype}{think_suffix}_{run_timestamp}"
        ),
    )
    os.makedirs(run_dir, exist_ok=True)
    return {
        "run_timestamp": run_timestamp,
        "run_dir": run_dir,
        "samples_jsonl": os.path.join(run_dir, "samples.jsonl"),
        "results_json": os.path.join(run_dir, "results.json"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="./cached_models")
    parser.add_argument("--local_files_only", action="store_true")

    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--max_prompt_tokens", type=int, default=None)
    parser.add_argument("--skip_warmup", action="store_true")
    parser.add_argument("--sample_start", type=int, default=0)
    parser.add_argument("--sample_end", type=int, default=None)
    parser.add_argument("--sample_indices", type=int, nargs="+", default=None)

    parser.add_argument("--compression_ratio", type=float, default=0.75)
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--kernel_size", type=int, default=7)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])

    parser.add_argument(
        "--think", action="store_true",
        help="Enable Qwen3 thinking mode (enable_thinking=True). No effect on non-Qwen models.",
    )
    parser.add_argument(
        "--think_max_new_tokens", type=int, default=4096,
        help="max_new_tokens used when --think is set (overrides dataset2maxlen.json / --max_new_tokens).",
    )

    parser.add_argument("--output_dir", type=str, default="benchmark_results")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    if args.think and "qwen" not in args.model_name.lower():
        print(f"WARNING: --think has no effect on {args.model_name!r} (only Qwen chat templates support enable_thinking).")

    model, tokenizer = build_model_and_tokenizer(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        dtype_str=args.dtype,
    )
    press = SnapKVPress(
        compression_ratio=args.compression_ratio,
        window_size=args.window_size,
        kernel_size=args.kernel_size,
    )

    with open(os.path.join(KIVI_CONFIG_DIR, "dataset2maxlen.json"), "r", encoding="utf-8") as f:
        dataset2maxlen = json.load(f)
    default_max_new_tokens = int(dataset2maxlen[DATASET_NAME])

    output_paths = build_output_paths(args)

    prompts = load_prompts(
        dataset_path=args.dataset_path,
        tokenizer=tokenizer,
        model_name=args.model_name,
        max_prompt_tokens=args.max_prompt_tokens,
        think=args.think,
    )
    if args.sample_indices is not None:
        selected_indices = set(args.sample_indices)
        prompts = [prompt for i, prompt in enumerate(prompts) if i in selected_indices]
    else:
        prompts = prompts[args.sample_start : args.sample_end]

    if not prompts:
        raise ValueError("No prompts remain after filtering")

    max_new_tokens = args.think_max_new_tokens if args.think else (args.max_new_tokens or default_max_new_tokens)

    results: List[SampleMetrics] = []
    scores: List[float] = []
    breakdown_detail: List[Dict[str, Any]] = []
    sample_records: List[Dict[str, Any]] = []
    skipped_oom = 0

    samples_jsonl_path = output_paths["samples_jsonl"]
    samples_file = open(samples_jsonl_path, "w", encoding="utf-8")

    def _write_sample_record(record: Dict[str, Any]) -> None:
        json.dump(record, samples_file, ensure_ascii=False)
        samples_file.write("\n")
        samples_file.flush()
        os.fsync(samples_file.fileno())

    if not args.skip_warmup:
        try:
            _ = benchmark_one(
                model, tokenizer, press, prompts[0]["prompt"],
                max_new_tokens=min(8, max_new_tokens),
                context_tokens=prompts[0]["context_tokens"],
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            if not is_cuda_oom(error):
                raise
            print("[warmup] CUDA OOM, skipping warmup and continuing with measured samples.")
            cleanup_cuda()

    for i, prompt_record in enumerate(prompts):
        prompt = prompt_record["prompt"]
        try:
            metrics = benchmark_one(
                model, tokenizer, press, prompt, max_new_tokens,
                context_tokens=prompt_record["context_tokens"],
            )
            results.append(metrics)
            score = score_prediction(metrics.pred, prompt_record["answers"])
            scores.append(score)
            breakdown_detail.append({
                "score": score,
                "difficulty": prompt_record.get("difficulty"),
                "length_category": prompt_record.get("length_category"),
            })
            sample_record = {
                "index": i,
                "max_new_tokens": max_new_tokens,
                "prompt": prompt,
                "answers": prompt_record["answers"],
                "score": score,
                "status": "ok",
                **asdict(metrics),
            }
            sample_records.append(sample_record)
            _write_sample_record(sample_record)
            out_tput_str = (
                f"{metrics.output_throughput_toks_per_s:.2f}"
                if metrics.output_throughput_toks_per_s is not None
                else "NA"
            )
            print(
                f"[{i + 1}/{len(prompts)}] "
                f"ttft={metrics.ttft_ms:.2f} ms | "
                f"out_tput={out_tput_str} tok/s | "
                f"prefill_cache={metrics.cache_size_mb:.2f} MB ({metrics.compressed_cache_tokens} tok) | "
                f"decode_cache={metrics.decode_cache_size_mb:.2f} MB ({metrics.decode_compressed_cache_tokens} tok) | "
                f"score={score:.2f}"
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            if not is_cuda_oom(error):
                raise
            skipped_oom += 1
            oom_record = {
                "index": i, "max_new_tokens": max_new_tokens, "prompt": prompt, "status": "oom",
            }
            sample_records.append(oom_record)
            _write_sample_record(oom_record)
            print(f"[{i + 1}/{len(prompts)}] CUDA OOM, skipping sample.")
            cleanup_cuda()

    samples_file.close()

    summary = summarize(results) if results else {}

    accuracy_overall = round(100 * sum(scores) / len(scores), 2) if scores else None
    breakdowns: Dict[str, Dict[str, float]] = {}
    for group_key in ("difficulty", "length_category"):
        buckets: Dict[str, List[float]] = {}
        for row in breakdown_detail:
            buckets.setdefault(str(row.get(group_key)), []).append(row["score"])
        breakdowns[group_key] = {k: round(100 * sum(v) / len(v), 2) for k, v in buckets.items()}

    payload = {
        "config": vars(args),
        "run_timestamp": output_paths["run_timestamp"],
        "dataset": DATASET_NAME,
        "max_new_tokens": max_new_tokens,
        "system_info": get_system_info(),
        "summary": summary,
        "accuracy": {
            "overall": accuracy_overall,
            **({"breakdowns": breakdowns} if breakdown_detail else {}),
        },
        "num_skipped_oom": skipped_oom,
        "samples": sample_records,
    }

    with open(output_paths["results_json"], "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\n=== ACCURACY ===")
    print(json.dumps(payload["accuracy"], indent=2))
    print(f"\nSaved per-sample results to: {output_paths['samples_jsonl']}")
    print(f"Saved summary results to: {output_paths['results_json']}")


if __name__ == "__main__":
    main()
