# KV-cache compression benchmarks

Compares KV-cache compression methods for long-context LLM inference on quality
(LongBench, LongBench-v2) and on performance (time to first token, throughput, cache
memory).

| ID | Method | Settings |
|----|--------|----------|
| FK | Full KV (baseline) | `k_bits=v_bits=16` |
| Q1 | KIVI 2-bit | group 32, residual 32 |
| Q2 | KIVI 2-bit | group 32, residual 128 |
| Q3 | KIVI 4-bit | group 32, residual 32 |
| Q4 | TurboQuant 4-bit | `bit_width=4` |
| E1 | SnapKV | ratio 0.75, window 64, kernel 5 |
| E2 | SnapKV | ratio 0.5, window 64, kernel 5 |
| M1 | CaM | ratio 0.75, target 1024, interval 32, merge budget 32 |
| M2 | CaM | ratio 0.75, target 1024, interval 512, merge budget 32 |

- **Study A:** LongBench, 13 datasets, on Llama-3.1-8B-Instruct and Qwen3-14B.
- **Study B:** LongBench-v2, 503 multiple-choice questions, on Qwen3-30B-A3B, with
  reasoning on and off.

## Reproduce the results

The original runs used NVIDIA A100-SXM4-40GB GPUs, Linux, and Python 3.10, with 40 GB of GPU
memory for Study A and 80 GB for Study B.

### 1. Install

`requirements.txt` is the only requirements file. The subprojects are installed with
`--no-deps` because their `pyproject.toml` pins conflict with it.

```bash
conda create -y -n kivi python=3.10     # run_benchmark.sh expects this env name
conda activate kivi
pip install -r requirements.txt

# flash-attn: required, and its build needs torch to be installed first.
pip install --no-build-isolation flash-attn==2.8.3

pip install --no-deps -e ./KIVI -e ./kvpress -e ./turboquant

conda create -y -n gcc12 -c conda-forge --override-channels gcc=12 gxx=12
PATH="$(conda info --base)/envs/gcc12/bin:$PATH" \
    pip install --no-deps --no-build-isolation -e ./KIVI/quant
```


### 2. Download models and datasets

Llama is gated, so accept its license on Hugging Face and run
`hf auth login` first.

```bash
hf download meta-llama/Llama-3.1-8B-Instruct   --cache-dir cached_models --revision 0e9e39f249a16976918f6564b8830bc894c89659
hf download Qwen/Qwen3-14B                     --cache-dir cached_models --revision 40c069824f4251a91eefaf281ebe4c544efd3e18
hf download Qwen/Qwen3-30B-A3B                 --cache-dir cached_models --revision ad44e777bcd18fa416d9da3bd8f70d33ebb85d39

hf download THUDM/LongBench --repo-type dataset --local-dir datasets/LongBench \
    --revision 5e628be450b7e67fb7ae6e201bd6d8f7056f7672
python datasets/prepare_longbench_v2.py
```

### 3. Run Study A

`run_benchmark.sh` runs each script from `/tmp` in the `kivi` env, so any path you pass
must be absolute. Add `--local_files_only` (`--local-files-only` for TurboQuant) to avoid
network access once the models are cached. Select a GPU with `CUDA_VISIBLE_DEVICES`.

```bash
DS="narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum lcc repobench-p"

for M in meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen3-14B; do
  KIVI="./run_benchmark.sh eval-kivi --model_name_or_path $M --dataset_names $DS --group_size 32"
  $KIVI --k_bits 16 --v_bits 16 --residual_length 32    # FK
  $KIVI --k_bits 2  --v_bits 2  --residual_length 32    # Q1
  $KIVI --k_bits 2  --v_bits 2  --residual_length 128   # Q2
  $KIVI --k_bits 4  --v_bits 4  --residual_length 32    # Q3

  ./run_benchmark.sh eval-turboquant --model $M --bit-widths 4 --tasks "${DS// /,}"   # Q4

  SNAP="./run_benchmark.sh eval-snapkv --model_name $M --dataset_names $DS --window_size 64 --kernel_size 5"
  $SNAP --compression_ratio 0.75   # E1
  $SNAP --compression_ratio 0.5    # E2

  CAM="./run_benchmark.sh eval-cam --model_name $M --dataset_names $DS --compression_ratio 0.75 --cam_target_size 1024 --cam_merge_budget 32"
  $CAM --cam_compression_interval 32    # M1
  $CAM --cam_compression_interval 512   # M2
done
```

Outputs go to timestamped folders in `benchmarking_accuracy/<method>/`. To score them,
move each run to `results_studyA/<ID>/<run folder>/` and run the
[commands below](#reproduce-the-study-a-tables-no-gpu). The scoring script takes the
configuration from the `<ID>` folder name and the model from the run folder name.

### 4. Run Study B

```bash
M=Qwen/Qwen3-30B-A3B

KIVI="./run_benchmark.sh kivi-longbenchv2 --model_name $M --group_size 32 --residual_length 128 --max_prompt_tokens 36800"
$KIVI --k_bits 16 --v_bits 16   # FK
$KIVI --k_bits 4  --v_bits 4    # Q3

./run_benchmark.sh snapkv-longbenchv2 --model_name $M --max_prompt_tokens 31500 \
    --compression_ratio 0.75 --window_size 64 --kernel_size 5   # E1

./run_benchmark.sh cam-longbenchv2 --model_name $M --max_prompt_tokens 31500 \
    --press_name cam_snapkv --compression_ratio 0.75 --compression_interval 32 \
    --target_size 1024 --merge_budget 32 --window_size 16 --kernel_size 5   # M1
```

- **Reasoning on:** add `--think` to each command. This raises the output limit to 4096
  tokens.
- **Splitting a run:** use `--sample_start` and `--sample_end` (end is exclusive).
- **Output:** `benchmark_results/<run>/samples.jsonl`, with a score for every question.

Mean scores per configuration, for one mode:

```bash
python export_scores.py --results results_studyB/reasoning_off --out scores_B_off.csv
```

### 5. Run the performance benchmarks

```bash
M=meta-llama/Llama-3.1-8B-Instruct   # or Qwen/Qwen3-14B
DS="narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum"

./run_benchmark.sh kivi --model_name $M --dataset_names $DS --k_bits 16 --v_bits 16 --group_size 32 --residual_length 32
./run_benchmark.sh snapkv --model_name $M --dataset_names $DS --compression_ratio 0.75 --window_size 32 --kernel_size 5
./run_benchmark.sh cam --model_name $M --dataset_names $DS --press_name cam_snapkv --compression_ratio 0.75 \
    --compression_interval 512 --target_size 2048 --merge_budget 32 --window_size 64 --kernel_size 5
./run_benchmark.sh turboquant --model_name $M --dataset_names $DS --bit_width 4
```

Settings differ between models and runs. To repeat a specific run, copy the values from
the `config` block at the top of its `results.json`:

| Model | KIVI residual length | CaM press (target / interval) |
|-------|---------------------|-------------------------------|
| Llama-3.1-8B-Instruct | 32 | `cam_snapkv` (2048 / 512), `cam_knorm` (1024 / 32) |
| Qwen3-14B | 128 (16-bit also at 32) | `cam_knorm` (2048 / 512 and 1024 / 32) |


## Reproduce the Study A tables (no GPU)

This regenerates every committed CSV from `results_studyA/`:

```bash
pip install pandas==2.3.3 numpy==2.2.6

python export_scores.py --results results_studyA --out scores_studyA.csv
python quality_ci.py scores_studyA.csv --baseline FK --scale 100 --out ci_studyA_vsFK.csv
python quality_ci.py scores_studyA.csv --baseline Q1 --scale 100 --out ci_Q2_vs_Q1.csv
python quality_ci.py scores_studyA.csv --baseline E1 --scale 100 --out ci_E2_vs_E1.csv
python quality_ci.py scores_studyA.csv --baseline M1 --scale 100 --out ci_M2_vs_M1.csv

```

## Vendored code

`KIVI/` and `kvpress/` are upstream projects with local changes, committed here as
ordinary files rather than submodules:

| Directory | Upstream | Base commit | License |
|-----------|----------|-------------|---------|
| `KIVI/` | <https://github.com/jy-yuan/KIVI> | `876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6` | MIT |
| `kvpress/` | <https://github.com/NVIDIA/kvpress> | `2173234256aeca24acf3e0da5840461cc05a6cbc` | Apache-2.0 |

Local changes add the `benchmarking_*` entry points and Llama-3.1 and Qwen model support.
They also modify `pred_long_bench.py` and `kvpress/evaluation/`. To see the exact changes,
diff against the base commits above.
