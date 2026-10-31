import argparse
import time

import torch
from transformers import AutoTokenizer, LlamaConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--cache_dir", type=str, default="./cached_models")
    parser.add_argument("--k_bits", type=int, default=2)
    parser.add_argument("--v_bits", type=int, default=2)
    parser.add_argument("--group_size", type=int, default=32)
    parser.add_argument("--residual_length", type=int, default=128)
    parser.add_argument("--use_flash", action="store_true")
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--prompt_length", type=int, default=160)
    parser.add_argument("--output_length", type=int, default=32)
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--warmup_runs", type=int, default=0)
    return parser.parse_args()


def build_model(args):
    config = LlamaConfig.from_pretrained(args.model_name_or_path)
    config.k_bits = args.k_bits
    config.v_bits = args.v_bits
    config.group_size = args.group_size
    config.residual_length = args.residual_length
    config.use_flash = args.use_flash
    config._flash_attn_2_enabled = args.use_flash

    if args.k_bits < 16 and args.v_bits < 16:
        if "llama-3.1" in args.model_name_or_path.lower():
            from models.llama_kivi_31 import LlamaForCausalLM_KIVI_31 as LlamaForCausalLM_KIVI
        else:
            from models.llama_kivi import LlamaForCausalLM_KIVI
        model = LlamaForCausalLM_KIVI.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path,
            config=config,
            cache_dir=args.cache_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
    else:
        from transformers import LlamaForCausalLM

        llama_kwargs = dict(
            pretrained_model_name_or_path=args.model_name_or_path,
            config=config,
            cache_dir=args.cache_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        if args.use_flash:
            llama_kwargs["attn_implementation"] = "flash_attention_2"
        model = LlamaForCausalLM.from_pretrained(**llama_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
    )
    return model.cuda().eval(), tokenizer


def build_inputs(tokenizer, batch_size, prompt_length):
    context = []
    for _ in range(batch_size):
        string = "t," * (prompt_length // 2)
        context.append(string[:-1])
    return tokenizer(context, return_tensors="pt").to("cuda")


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cleanup_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def timed_call(fn, num_repeats):
    total_ms = 0.0
    for _ in range(num_repeats):
        synchronize()
        start = time.perf_counter()
        result = fn()
        synchronize()
        total_ms += (time.perf_counter() - start) * 1000
        del result
        cleanup_cuda()
    return total_ms / num_repeats


def measure_prefill_and_ttft(model, input_ids, attention_mask=None, num_warmups=2, num_repeats=10):
    prefill_times = []
    first_decode_times = []
    ttft_times = []

    def run_prefill_and_first_decode():
        prefill_kwargs = {
            "input_ids": input_ids,
            "use_cache": True,
        }
        if attention_mask is not None:
            prefill_kwargs["attention_mask"] = attention_mask

        out = model(**prefill_kwargs)
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        decode_kwargs = {
            "input_ids": next_token,
            "past_key_values": out.past_key_values,
            "use_cache": True,
        }
        if attention_mask is not None:
            next_attention_mask = torch.cat(
                [
                    attention_mask,
                    attention_mask.new_ones((attention_mask.shape[0], 1)),
                ],
                dim=1,
            )
            decode_kwargs["attention_mask"] = next_attention_mask

        return out, decode_kwargs

    with torch.no_grad():
        for _ in range(num_warmups):
            out, decode_kwargs = run_prefill_and_first_decode()
            first_out = model(**decode_kwargs)
            del out, first_out, decode_kwargs
            cleanup_cuda()
        synchronize()

        for _ in range(num_repeats):
            synchronize()
            t0 = time.perf_counter()

            out, decode_kwargs = run_prefill_and_first_decode()

            synchronize()
            t1 = time.perf_counter()

            first_out = model(**decode_kwargs)

            synchronize()
            t2 = time.perf_counter()

            prefill_times.append((t1 - t0) * 1000.0)
            first_decode_times.append((t2 - t1) * 1000.0)
            ttft_times.append((t2 - t0) * 1000.0)
            del out, first_out, decode_kwargs
            cleanup_cuda()

    avg_prefill_ms = sum(prefill_times) / len(prefill_times)
    avg_first_decode_ms = sum(first_decode_times) / len(first_decode_times)
    avg_ttft_ms = sum(ttft_times) / len(ttft_times)
    batch_prompt_tokens = input_ids.shape[0] * input_ids.shape[1]

    return {
        "prefill_ms": avg_prefill_ms,
        "first_decode_ms": avg_first_decode_ms,
        "ttft_ms": avg_ttft_ms,
        "prefill_tok_per_s": batch_prompt_tokens / avg_prefill_ms * 1000.0,
        "decode_tok_per_s": input_ids.shape[0] / avg_first_decode_ms * 1000.0,
    }


def main():
    args = parse_args()
    model, tokenizer = build_model(args)
    inputs = build_inputs(tokenizer, args.batch_size, args.prompt_length)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")

    print(
        f"bs: {args.batch_size}, seqlen: {input_ids.shape[1]}+{args.output_length}\n"
        f"model: {args.model_name_or_path}"
    )

    with torch.no_grad():
        for _ in range(args.warmup_runs):
            warmup_out = model(**inputs, use_cache=True)
            del warmup_out
            cleanup_cuda()
        synchronize()

        cleanup_cuda()
        torch.cuda.reset_peak_memory_stats()

        latency_metrics = measure_prefill_and_ttft(
            model,
            input_ids,
            attention_mask=attention_mask,
            num_warmups=args.warmup_runs,
            num_repeats=args.num_repeats,
        )
        total_gen_ms = timed_call(
            lambda: model.generate(**inputs, max_new_tokens=args.output_length, do_sample=False),
            args.num_repeats,
        )

        used_mem = torch.cuda.max_memory_allocated()

    print(f"prefill latency: {latency_metrics['prefill_ms']:.2f} ms")
    print(f"first decode latency: {latency_metrics['first_decode_ms']:.2f} ms")
    print(f"time to first token: {latency_metrics['ttft_ms']:.2f} ms")
    print(f"prefill throughput: {latency_metrics['prefill_tok_per_s']:.2f} tok/s")
    print(f"decode throughput: {latency_metrics['decode_tok_per_s']:.2f} tok/s")
    print(f"used time: {total_gen_ms:.2f} ms")
    print(f"peak mem: {used_mem / 1024 ** 3:.2f} GB")


if __name__ == "__main__":
    main()
