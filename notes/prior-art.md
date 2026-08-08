# Prior-art search & repositioning record

**Date:** 2026-08-08
**Status:** resolves SRS §12 Q4 (the existential gate) and week1.md item 1.
**Rule invoked:** SRS §12 — "If [a matched-budget comparison exists], pivot." This file *is* the pivot.

This document is committed **before** any experiment is run. It is the pre-registration signal
(SRS §9) and it timestamps that the repositioning below was a deliberate response to the
literature, not a post-hoc rationalization of whatever the data showed.

---

## 1. The existential question and its answer

> SRS §12 Q4 / week1.md: *Does a matched-budget head-to-head comparison of KV eviction methods
> already exist?*

**Yes. Two papers, both verified by fetching the arXiv abstract (and, for the first, the full
HTML). The planned RQ2 and RQ3 are already published.**

### 1.1 The direct hit — RQ2 and RQ3

**Luo, Liang, Xuan. "How Query Visibility Changes KV-Cache Compression Rankings: A Matched-Budget
Audit." arXiv:2607.11942, 11 Jul 2026. 12 pp, 5 figures.** (cs.LG)

Verified from the paper's own HTML:

- **Budget:** uniform compression ratio `r ∈ {0.25, 0.5, 0.75, 0.9}` = fraction of tokens
  retained. Coverage enforced programmatically; "331 OOM holes detected and refilled." **Not bytes.**
- **Arms:** SnapKV, H2O, TOVA, ExpectedAttention, AdaKV, KeyDiff (6 methods) vs
  Random, Knorm, StreamingLLM (3 trivial baselines). Compared against "best-of-3 trivial" per cell.
- **Models:** Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, DeepSeek-R1-Distill-Qwen-7B.
- **Benchmarks:** RULER-8192 (13 subtasks, 650 instances) + LongBench (16 English tasks × 50).
- **Manipulation:** *binary* query-visibility toggle. Query-aware = question visible to the
  scoring pass; query-agnostic = context compressed first, question appended after eviction.
  **No sweep over query position.**
- **Headline:** "SnapKV ... loses to 'keep the start and the recent window' on average (−0.066)"
  under the query-agnostic (realistic-deployment) protocol. Degradation is proportional to how
  much query information leaks into the scoring signal — SnapKV drops 0.198 (question sits in its
  64-token observation window); KeyDiff drops 0.011 (no query in its score).
- **Measures:** task accuracy only. **No memory, no latency, no throughput.** Total compute ~US$15.

This is SRS **RQ2** (query-aware methods exploit question-at-end) and **RQ3** (trivial baselines
close the gap), executed at larger scale than the SRS planned, one month before the SRS was drafted.

### 1.2 Second hit — RQ3

**Garcia. "Protection Is (Nearly) All You Need: Structural Protection Dominates Scoring in
Globally Capped KV Eviction." arXiv:2605.18053, 18 May 2026. 38 pp.**

- Arms include Random and LRU (recency) as first-class policies (also H2O, SnapKV, StreamingLLM,
  Ada-KV, QUEST). Budget in absolute token counts C ∈ {256, 512, 2048}.
- Finding: without structural protection at prompt boundaries all seven collapse (F1 ≤ 0.064);
  reserving 10% of cache at boundaries recovers 69–90% of quality; with protection, attention
  scoring buys only 0.011–0.021 F1 over LRU. "Protection dominates; scoring differences are
  secondary."

This is SRS **RQ3** (trivial baselines are competitive).

### 1.3 Verdict

RQ2 and RQ3 are claimed. Per the SRS rule, the project pivots. It does **not** die: the pivot
target below is named by the scooping paper's own limitations section.

---

## 2. What is still open — the pivot

**The entire literature budgets in tokens (or a token-fraction "compression ratio") and reports
the result as memory savings. Tokens are not bytes, and the gap is method-dependent.**

Evidence that this axis is unclaimed:

- **2607.11942 limitations, verbatim:** *"Uniform ratio budgets only; adaptive allocation
  unexplored"* and *"KVzip, Compactor, PyramidKV absent."* It measures no memory at all.
- **2605.18053:** token counts, not bytes.
- **NVIDIA KVPress** (the field's de facto tool): prefill knob is `compression_ratio` = fraction
  of *tokens* dropped; decode knob is `target_size` = token count. No byte budget.
- **longctx_bench (EMNLP'24, arXiv:2407.01527), SCBench (ICLR'25, arXiv:2412.10319):**
  ratio / lifecycle framing, not bytes.
- Nearest published gesture: **Agrawal & Mayer, arXiv:2607.05399** — "compression ratio alone is
  a poor predictor of end-to-end performance" — but ratio-based, and only SnapKV among the token
  eviction methods.

### 2.1 The claim is already partly verified by source inspection (not just hypothesized)

1. **`AdaKVPress` in NVIDIA KVPress does not free memory.** `kvpress/presses/adakv_press.py`
   returns `keys, values` **unmodified** and stores `module.masked_key_indices` for the attention
   patch (`attention_patch.py`) to mask downstream. Tensors stay full-size. Realized bytes = full
   cache; reported `compression_ratio` = large. Inherited by `CriticalAdaKVPress` and head-wise
   variants.
   - *To re-verify in Phase 1 against a pinned commit + read attention_patch.py to confirm the
     mask never triggers a physical slice.*
2. **Head-level allocation cannot save GQA memory even if implemented physically.** The unit of KV
   memory under GQA is the (layer, KV-group) cell; query heads in a group share K/V, so head-level
   budget concentration frees nothing without paged/ragged storage — which is exactly what
   KV-Compress (arXiv:2410.00161) had to build.
3. **Peak-memory overhead is invisible to ratio accounting.** `KVComposePress` makes an extra full
   pass, transiently allocating ~2× context KV at prefill — "compresses" while raising the peak
   that decides whether the job fits (SRS §4.5 confound 2).

### 2.2 Framing discipline

This is a **protocol critique, not an accusation.** Masking is a legitimate way to study a
method's *accuracy*, and KVPress ships a `speed_and_memory` notebook. The defect is that the
literature does not distinguish **simulation-faithful** from **memory-faithful** implementations
and reports both as compression. The contribution is that taxonomy + its measurement. Wording that
implies NVIDIA misrepresents its tool is unfair and a reviewer liability.

---

## 3. Pre-registered predictions for Phase 2 (recorded before running)

The byte-accounting audit (nominal ratio → realized bytes, one line per method, one model, one
context length) is predicted to show:

- **AdaKVPress: flat at ~1.0** (100% of full cache) across all nominal ratios. *This is the
  positive control — if the instrument does not reproduce it, the instrument is broken.*
- **SnapKV, StreamingLLM: diverge upward** from the "ratio promises" diagonal by the byte size of
  the observation window (SnapKV) and sink + recency allocation (StreamingLLM) that sit outside
  the counted budget.
- **Random, Knorm, TOVA, KeyDiff: near the diagonal** (memory-faithful).
- **ThinKPress: off-axis** — compresses channels, so a token-fraction ratio is not a memory
  statement about it at all.
- **Kendall τ** between the token-matched ranking and the byte-matched ranking is predicted `< 1`
  at one or more budget levels (H1 restated in byte terms).

Any deviation from these predictions is itself reportable (SRS NFR4: honest failure).

---

## 4. Harness decision (week1.md item 2)

**Fork NVIDIA/kvpress** (Apache-2.0, modern HF `Cache` API, forward-hook based, ships SnapKV /
PyramidKV / StreamingLLM / TOVA / AdaKV / KeyDiff / Random / ObservedAttention / ThinK /
ExpectedAttention + a RULER/LongBench eval loop). **Do not** fork KVCache-Factory (monkey-patched,
pinned to `transformers==4.44.2`). Delta to build: the byte instrument + `ByteBudgetPress` wrapper.

Note: KVPress has no `H2O` by name — `ObservedAttentionPress` (accumulated observed attention) is
the closest; a true H2O may need to be added.

---

## 5. Positioning sentence for the paper

> Concurrent work (Luo et al., arXiv:2607.11942) shows the field's method rankings are unstable
> under *query visibility*. We show they are also unstable under the *resource unit* the field
> measures them in: re-pricing the standard comparison in realized bytes rather than token
> fractions changes the ordering, and in the limiting case reveals methods that the standard
> tooling reports as compressing while they free no memory at all.

---

## Sources

- [arXiv:2607.11942](https://arxiv.org/abs/2607.11942) · [arXiv:2605.18053](https://arxiv.org/abs/2605.18053)
- [NVIDIA/kvpress](https://github.com/NVIDIA/kvpress) · [adakv_press.py](https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/adakv_press.py)
- [KV-Compress arXiv:2410.00161](https://arxiv.org/abs/2410.00161)
- [longctx_bench arXiv:2407.01527](https://arxiv.org/abs/2407.01527) · [SCBench arXiv:2412.10319](https://arxiv.org/abs/2412.10319) · [arXiv:2607.05399](https://arxiv.org/abs/2607.05399)
