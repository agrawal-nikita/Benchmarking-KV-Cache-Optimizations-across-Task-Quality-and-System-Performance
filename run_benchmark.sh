#!/usr/bin/env bash
# Unified benchmark runner — single conda env "kivi".
#
# Throughput / memory benchmarking:
#   ./run_benchmark.sh kivi        [--model_name ...] [--dataset_names ...] [...]
#   ./run_benchmark.sh snapkv      [--model_name ...] [--compression_ratio ...] [...]
#   ./run_benchmark.sh cam         [--model_name ...] [--press_name ...] [...]
#   ./run_benchmark.sh turboquant  [--model_name ...] [--bit_width ...] [...]
#
# LongBench accuracy evaluation (saves to benchmarking_accuracy/):
#   ./run_benchmark.sh eval-kivi       --model_name_or_path meta-llama/Llama-3.1-8B-Instruct [--k_bits 2] [--v_bits 2] [...]
#   ./run_benchmark.sh eval-snapkv     --model_name meta-llama/Llama-3.1-8B-Instruct [--compression_ratio 0.3] [...]
#   ./run_benchmark.sh eval-cam        --model_name meta-llama/Llama-3.1-8B-Instruct [--compression_ratio 0.3] [...]
#   ./run_benchmark.sh eval-turboquant --model meta-llama/Llama-3.1-8B-Instruct [--bit-widths 3,4,baseline] [...]
#
# LongBench-v2 accuracy evaluation (multiple-choice, longer contexts) works the
# same way as v1 — first cache it locally once with
# `python datasets/prepare_longbench_v2.py`, then point eval-kivi/eval-snapkv/eval-cam
# at it:
#   ./run_benchmark.sh eval-kivi --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
#       --dataset_path datasets/LongBench-v2 --dataset_names longbench_v2 --k_bits 2 --v_bits 2
#
# LongBench-v2 performance benchmarking (dedicated scripts, saves to benchmark_results/):
#   ./run_benchmark.sh kivi-longbenchv2   --model_name_or_path meta-llama/Llama-3.1-8B-Instruct [--k_bits 2] [--think] [...]
#   ./run_benchmark.sh snapkv-longbenchv2 --model_name meta-llama/Llama-3.1-8B-Instruct [--compression_ratio 0.75] [--think] [...]
#   ./run_benchmark.sh cam-longbenchv2    --model_name meta-llama/Llama-3.1-8B-Instruct [--press_name cam_snapkv] [--think] [...]
# These are separate, dedicated scripts (not the v1 kivi/snapkv/cam ones above) because
# LongBench-v2's context lengths need different context_length_buckets (0-32k/32k-128k/128k+
# instead of v1's 0-4k/4k-8k/8k+). Each reports TTFT, throughput, KV cache footprint at both
# prefill and end-of-decode (cache_size_mb / decode_cache_size_mb), and accuracy in one run.
# --dataset_path defaults to datasets/LongBench-v2 already (an absolute path computed from
# the script's own location, so — unlike the other commands here — you don't need to pass
# an absolute --dataset_path yourself to work around run_kivi_env()'s `cd /tmp`).
#
# All methods run inside conda env "kivi" (Python 3.10).
# kvpress and turboquant are installed there in editable mode alongside KIVI.
#
# Common flags (benchmarking):
#   --dataset_names narrativeqa gov_report qasper ...   (default: all 21 LongBench datasets)
#   --dataset_path  /path/to/LongBench                  (default: local datasets/LongBench)
#   --model_name    meta-llama/Llama-3.1-8B-Instruct
#   --cache_dir     ./cached_models
#   --local_files_only
#   --output_dir    benchmark_results
#   --sample_start / --sample_end / --sample_indices

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD="${1:-}"

if [[ -z "$METHOD" ]]; then
    echo "Usage: $0 <kivi|snapkv|cam|turboquant> [args...]"
    exit 1
fi
shift

# Run a script in the kivi conda env from /tmp so the repo root is not on sys.path.
# This prevents subproject folders at the repo root from shadowing their editable installs.
# PYTHONNOUSERSITE=1 prevents ~/.local ABI-mismatch conflicts.
run_kivi_env() {
    local script="$1"; shift
    cd /tmp
    PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        conda run --no-capture-output -n kivi python "$script" "$@"
}

case "$METHOD" in
    kivi)
        run_kivi_env \
            "$REPO_ROOT/KIVI/benchmarking_kivi.py" \
            --cache_dir    "$REPO_ROOT/cached_models" \
            --dataset_path "$REPO_ROOT/datasets/LongBench" \
            --output_dir   "$REPO_ROOT/benchmark_results" \
            "$@"
        ;;

    snapkv)
        run_kivi_env \
            "$REPO_ROOT/kvpress/benchmarking_snapkv_kvpress.py" \
            --cache_dir    "$REPO_ROOT/cached_models" \
            --dataset_path "$REPO_ROOT/datasets/LongBench" \
            --output_dir   "$REPO_ROOT/benchmark_results" \
            "$@"
        ;;

    cam)
        run_kivi_env \
            "$REPO_ROOT/kvpress/benchmarking_cam_kvpress.py" \
            --cache_dir    "$REPO_ROOT/cached_models" \
            --dataset_path "$REPO_ROOT/datasets/LongBench" \
            --output_dir   "$REPO_ROOT/benchmark_results" \
            "$@"
        ;;

    turboquant)
        run_kivi_env \
            "$REPO_ROOT/turboquant/benchmarking_turboquant.py" \
            --cache_dir    "$REPO_ROOT/cached_models" \
            --dataset_path "$REPO_ROOT/datasets/LongBench" \
            --output_dir   "$REPO_ROOT/benchmark_results" \
            "$@"
        ;;

    kivi-longbenchv2)
        run_kivi_env \
            "$REPO_ROOT/KIVI/benchmarking_kivi_longbench_v2.py" \
            --cache_dir    "$REPO_ROOT/cached_models" \
            --output_dir   "$REPO_ROOT/benchmark_results" \
            "$@"
        ;;

    snapkv-longbenchv2)
        run_kivi_env \
            "$REPO_ROOT/kvpress/benchmarking_snapkv_kvpress_longbench_v2.py" \
            --cache_dir    "$REPO_ROOT/cached_models" \
            --output_dir   "$REPO_ROOT/benchmark_results" \
            "$@"
        ;;

    cam-longbenchv2)
        run_kivi_env \
            "$REPO_ROOT/kvpress/benchmarking_cam_kvpress_longbench_v2.py" \
            --cache_dir    "$REPO_ROOT/cached_models" \
            --output_dir   "$REPO_ROOT/benchmark_results" \
            "$@"
        ;;

    eval-kivi)
        run_kivi_env \
            "$REPO_ROOT/KIVI/pred_long_bench.py" \
            --dataset_path "$REPO_ROOT/datasets/LongBench" \
            --cache_dir    "$REPO_ROOT/cached_models" \
            --output_dir   "$REPO_ROOT/benchmarking_accuracy/kivi" \
            "$@"
        ;;

    eval-snapkv)
        run_kivi_env \
            "$REPO_ROOT/kvpress/benchmarking_eval_longbench.py" \
            --press_name snapkv \
            --dataset_path "$REPO_ROOT/datasets/LongBench" \
            --cache_dir    "$REPO_ROOT/cached_models" \
            --output_dir   "$REPO_ROOT/benchmarking_accuracy/snapkv" \
            "$@"
        ;;

    eval-cam)
        run_kivi_env \
            "$REPO_ROOT/kvpress/benchmarking_eval_longbench.py" \
            --press_name cam \
            --dataset_path "$REPO_ROOT/datasets/LongBench" \
            --cache_dir    "$REPO_ROOT/cached_models" \
            --output_dir   "$REPO_ROOT/benchmarking_accuracy/cam" \
            "$@"
        ;;

    eval-turboquant)
        run_kivi_env \
            "$REPO_ROOT/turboquant/benchmarks/eval_longbench.py" \
            --dataset-path "$REPO_ROOT/datasets/LongBench" \
            --cache-dir    "$REPO_ROOT/cached_models" \
            --output_dir   "$REPO_ROOT/benchmarking_accuracy/turboquant" \
            "$@"
        ;;

    *)
        echo "Unknown method: '$METHOD'"
        echo "Benchmarking:            kivi, snapkv, cam, turboquant"
        echo "Accuracy eval:           eval-kivi, eval-snapkv, eval-cam, eval-turboquant"
        echo "LongBench-v2 benchmarks: kivi-longbenchv2, snapkv-longbenchv2, cam-longbenchv2"
        exit 1
        ;;
esac
