# Running kv-bench

## Setup (once)

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
git clone --depth 1 https://github.com/NVIDIA/kvpress.git vendor/kvpress
uv pip install -e ./vendor/kvpress matplotlib pyyaml
```

## Sanity: the byte instrument must pass before trusting any number

```bash
PYTHONPATH=src python -m kvbench.selftest
```

Expected: all 3 controls PASS, including the masking positive control (full payload, meta > 0).

## Phase 2 — the byte audit (no accuracy eval; the motivating figures)

```bash
PYTHONPATH=src python -m kvbench.run --config configs/pilot.yaml --results results
python scripts/fig1_mispricing.py --results results --out reports/figures/fig1_mispricing.pdf
```

Produces Fig 1 (realized bytes vs nominal ratio) and the taxonomy. AdaKV measures at ~1.0–1.1
(simulation-only); memory-faithful methods lie on the diagonal.

## Phase 3 — accuracy re-ranking at matched bytes

Local smoke test (0.5B, floor-heavy accuracy — proves the pipeline):

```bash
PYTHONPATH=src python -m kvbench.eval --config configs/eval_pilot.yaml --results results
```

**The real run — on the Ubuntu / CUDA box** (8B won't fit in 16 GB locally):

```bash
PYTHONPATH=src python -m kvbench.eval --config configs/main.yaml --device cuda
```

`configs/main.yaml` uses Qwen2.5-3B and 7B at ctx 2048/4096/8192, 25 needles per cell. For
Llama-3.1-8B: accept the license on HuggingFace, `huggingface-cli login`, then uncomment it.

Then regenerate the paper figures/tables:

```bash
python scripts/fig2_reranking.py  --results results --out reports/figures/fig2_reranking.pdf
python scripts/table1_rank.py     --results results --out reports/figures/table1_rank.tex
```

## Build the paper

```bash
cd reports && make paper        # verifies bib, regenerates figures, runs latexmk
```

Or upload `reports/` to Overleaf (figures are committed, so it compiles as-is).

## Notes on portability

- Device is auto-selected (CUDA > MPS > CPU); force with `--device`.
- `observed_attention` (H2O surrogate) requires eager attention — the runner loads an eager model
  for it automatically. This incompatibility with SDPA/Flash is recorded as a systems finding.
- Every cell writes one JSON to `results/`; runs are resumable (existing cells are skipped).
- Failures are recorded as `{"failed": true, ...}`, never interpolated (SRS NFR4).
