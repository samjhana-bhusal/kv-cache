# reports/ — the paper and its build

**Working title:** *Bytes, Not Ratios: What KV Cache Compression Benchmarks Actually Measure.*

## What is here

| File | Role |
|---|---|
| `paper.tex` | The paper. Portable `article` skeleton — compiles anywhere with a LaTeX toolchain; swap `\documentclass` for the venue style before submission. |
| `related_work.tex` | `\input` into the paper; also reads standalone. Organized by mechanism family. |
| `refs.bib` | Bibliography. Every arXiv entry is machine-verifiable via `scripts/verify_bib.py`. |
| `Makefile` | `make paper` regenerates figures from `../results/` then builds the PDF. |
| `figures/` | Generated PDFs only — never hand-edited, never committed with fake data. |

## Building

```bash
# Draft build (placeholders allowed where results/ is still empty):
cd reports && make draft

# Real build (fails loudly if any figure's result cells are missing):
cd reports && make paper
```

`make paper` runs, in order: `verify_bib.py` (every citation resolved against arXiv) →
`scripts/fig*.py` (each reads `../results/*.json` and **exits non-zero rather than emit a
placeholder plot** if its data is missing) → `latexmk`.

## The one rule

**No number in this paper is typed by hand, and no figure is drawn without data.** Figures render
from `../results/*.json` via `scripts/`, or `paper.tex` shows a loud red "MISSING DATA FIGURE" box.
Quantitative claims in the abstract/body are bracketed placeholders until the sweep fills them.
This is the SRS §4.4 reproducibility discipline, enforced by the build rather than by good
intentions.

## Status

**Feature-complete for a workshop submission.** All body text is real prose (no bracketed
placeholders); both included figures and the table render from real, committed `results/*.json`;
all 42 bibliography entries are machine-verified against arXiv (`scripts/verify_bib.py`, clean).

- [x] Related Work (9 method families, verified citations) and full Introduction/Conclusion.
- [x] Fig 1 (byte audit): Qwen2.5-0.5B **and** 3B, MPS + Kaggle T4. Taxonomy confirmed at two scales.
- [x] Fig 2 + Table 1 (re-ranking): real **RULER** (Qwen2.5-3B, ctx4096, 50 items/cell, 13 tasks) —
      not the synthetic pilot. AdaKV and ThinK byte-infeasible at every level; StreamingLLM beats
      SnapKV at every matched byte budget.
- [x] Bibliography verified end-to-end (42/42 entries resolve against the arXiv API).
- [ ] **Explicitly out of scope, documented in Limitations, not silently missing:** per-method byte
      decomposition and decode-throughput-at-matched-bytes. Instrumentation stubs exist
      (`scripts/fig3_decomposition.py`, `scripts/fig5_latency.py`) but are not in `make paper`'s
      required build and are not referenced by `paper.tex` — they need fields
      (`byte_decomposition`, `decode_tokens_per_s`) no current result cell carries. Adding them is
      a mechanical extension, not a rewrite.
- [ ] 7–8B model, 8192+ context, LongBench: named as future work in the paper, not attempted.

See `../notes/prior-art.md` for why the paper is framed around bytes, `../RUN.md` for how to run,
and the plan file for the phase-by-phase schedule.
