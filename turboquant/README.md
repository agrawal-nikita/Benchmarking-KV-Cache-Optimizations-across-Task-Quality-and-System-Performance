# TurboQuant

A TurboQuant KV cache for Hugging Face Transformers, installed as the `turboquant_kv` package.

- The cache integration is built directly on `DynamicCache`.
- The quantization math is kept from the original TurboQuant project.

## Usage

From this folder, run:

```bash
PYTHONPATH=. python -m benchmarks.local
```
