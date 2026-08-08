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

- [x] Related Work written from verified citations (Phase 0/5).
- [x] Figure pipeline wired and fail-loud (`scripts/fig*.py`, `_figlib.py`).
- [ ] `results/` populated — needs the harness (`src/kvbench/`) and a compute environment.
- [ ] Figures generated; placeholders replaced.
- [ ] Quantitative claims filled.

See `../notes/prior-art.md` for why the paper is framed around bytes, and the plan file for the
phase-by-phase schedule.
