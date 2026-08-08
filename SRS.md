# SRS — Matched-Budget Evaluation of KV Cache Eviction Methods

**Status:** Draft v2, 2026-08-08 — **repositioned after prior-art search.** See [notes/prior-art.md](notes/prior-art.md).
(Draft v1, 2026-08-06, predates this repo's first commit and was edited in place into v2; the
change of spine is documented in notes/prior-art.md rather than a v1 commit.)
**Target:** NeurIPS 2026 workshop (MLForSys / AXIOM, Aug 28–29) → ICLR 2027 (Sep 25); arXiv preprint by Nov 20; listable for Fall 2027 PhD applications.

---

## 1. Purpose

KV cache eviction methods (StreamingLLM, H2O, SnapKV, PyramidKV, …) each report substantial wins, but under **mismatched budgets, models, tasks, and baselines**. Concurrent 2026 work (arXiv:2607.11942, arXiv:2605.18053) now runs them head-to-head under matched *token* budgets — but budgets in tokens and reports the result as memory savings. **No published work fixes the budget in realized bytes**, and the field's standard tool (NVIDIA KVPress) contains implementations that report large compression ratios while freeing no memory at all (notes/prior-art.md).

This project builds the byte-accounting protocol, runs it, and reports whether the token-matched ordering survives when the resource unit is corrected to bytes.

The deliverable is a paper **whether or not the ordering changes**. A null result ("the published ordering holds under matched conditions") is a useful, publishable contribution — this is the property that makes the project low-risk on a fixed timeline.

### 1.1 Non-goals

- Inventing a new eviction method. Explicitly out of scope. (If a novel method falls out of the analysis, that is a *bonus*, not a requirement.)
- KV **quantization** methods (KIVI, KVQuant). Different axis; note as related work, do not evaluate.
- Training or fine-tuning anything.
- Multi-GPU / distributed serving.

---

## 2. Research questions

**Repositioning note (v2).** The prior-art search (notes/prior-art.md) found that the original
spine, RQ2 (query-aware methods exploit question-at-end), was published one month before this SRS
was drafted: Luo et al., *"How Query Visibility Changes KV-Cache Compression Rankings,"*
[arXiv:2607.11942](https://arxiv.org/abs/2607.11942), 11 Jul 2026 — which also settles RQ3
(trivial baselines close the gap), as does Garcia [arXiv:2605.18053](https://arxiv.org/abs/2605.18053).
Per the §12 pivot rule, **RQ1 is now the spine, sharpened.** RQ2/RQ3 become *replication of
concurrent work at matched bytes*, not primary claims.

The paper's spine is now: **the field budgets compression in tokens and reports it as memory
savings — and those are not the same quantity.** If only one thing gets done, produce the
byte-accounting audit (RQ1a below).

### RQ1a — What do "compression ratios" actually cost in bytes? ⭐ (new spine)
Every benchmark and the field's standard tool (NVIDIA KVPress) budgets by token fraction. But
methods differ in what the fraction excludes (sinks, recency and observation windows, per-layer/
per-head allocation tables, score accumulators) and whether they compress along the token axis at
all (ThinK prunes channels). Some implementations do not shrink tensors: KVPress's `AdaKVPress`
returns K/V unmodified and masks downstream — reported ratio large, realized bytes unchanged
(verified by source inspection; notes/prior-art.md §2.1). Head-level allocation cannot save GQA
memory in principle (shared K/V per group).

**H1a — taxonomy:** presses partition into *memory-faithful* (tensors shrink), *simulation-only*
(masked, realized = full cache), and *incommensurable* (off the token axis). Measurable directly,
no accuracy eval.

**H1b — re-ranking:** when budget is fixed as realized total KV bytes, the token-matched method
ranking changes (Kendall τ < 1) at one or more budget levels.

**H1c — systems:** simulation-only / masked-ragged methods that free no bytes also yield no decode
speedup, while being ranked as if they saved both.

### RQ2 — Query position at matched bytes (replication, secondary; ICLR-version only)
Concurrent work (2607.11942) toggles query visibility *binary* at matched *tokens*. We ask whether
the effect survives at matched *bytes*, and extend the toggle to a position sweep
(start/middle/end) with a full-cache control separating compression-induced position bias from the
model's native lost-in-the-middle prior. **Cited as concurrent work, never claimed as ours.**
Droppable; not in the workshop paper.

### RQ3 — Trivial baselines (replication, folded into RQ1b)
Random and recency are first-class arms in the byte-matched sweep, replicating 2607.11942 /
2605.18053 under the corrected resource unit. Not an independent contribution.

### RQ4 — MHA → GQA transfer (retained, droppable)
**H4:** Ranking established on an MHA model does not transfer to a GQA model at equivalent byte
budget. GQA's shared-KV structure is also *why* head-level allocation cannot save memory (RQ1a),
so this question is now entangled with the spine rather than separate from it.

---

## 3. Methods under evaluation

Verify every citation and implementation detail against the papers/repos before building — the summaries below are from memory and must not be trusted for implementation.

| Method | Family | Query-aware? | Key idea |
|---|---|---|---|
| **Full cache** | — | — | Upper-bound control |
| **Random** | trivial | no | Uniformly evict to budget. **Required baseline.** |
| **Recency / sliding window** | trivial | no | Keep last *k*. **Required baseline.** |
| **StreamingLLM** | sink+window | no | Attention sinks (first few tokens) + recent window |
| **H2O** | score-based | no | Evict by accumulated attention score ("heavy hitters") |
| **SnapKV** | score-based | **yes** | Observation window at prompt end selects prefix tokens |
| **PyramidKV** | score-based | **yes** | Per-layer budget allocation (more in lower layers) |

Optional stretch (only if ahead of schedule): TOVA, Scissorhands, FastGen.

The **query-aware column is the experimental grouping for RQ2.** Getting this classification right matters more than method count.

---

## 4. Experimental design

### 4.1 Controlled (held fixed)
- Model weights and quantization
- Decoding: **greedy** (temperature 0) for determinism
- Prompt text, few-shot examples, formatting
- Random seeds (fixed and recorded)
- Budget expressed in **bytes**, not token fraction

### 4.2 Independent variables
| Variable | Levels |
|---|---|
| Method | 7 above |
| Byte budget | ~5 levels, spanning aggressive → mild (e.g. equivalent of 128 / 256 / 512 / 1024 / 2048 tokens at full precision) |
| Context length | 2k, 4k, 8k, 16k, (32k if memory allows) |
| **Question position** | **start / end** ← RQ2 manipulation |
| Model | one MHA, one GQA |

### 4.3 Dependent variables
- **Primary:** downstream task accuracy (LongBench subsets; Needle-in-a-Haystack)
- **Secondary:** peak memory, **measured not analytically derived**
- **Secondary:** decode latency, tokens/sec, time-to-first-token
- **Tertiary:** perplexity — record it, but treat as a weak proxy and say so; PPL is known to correlate poorly with long-context task performance

### 4.4 Statistical protocol
Carry over the discipline from the RMI project — it is the differentiator:
- ≥3 runs per cell where any nondeterminism exists; report **mean ± stddev**
- Report **all** cells, including ones where the finding is null
- Pre-register the hypotheses (H1–H4) in the repo before running the main sweep — a timestamped commit is enough, and it is a strong credibility signal
- Every number in the paper must be reproducible from a script in this repo

### 4.5 Known confounds to control explicitly
1. **Observation-window leakage** — the RQ2 manipulation
2. **Prefill vs decode accounting** — some methods do not reduce *peak* prefill memory; report peak and steady-state separately
3. **Budget ambiguity for per-layer methods** — normalize by total bytes
4. **GQA vs MHA** — RQ4
5. **Tokenizer differences** across models changing effective context

---

## 5. Functional requirements

### FR1 — Cache backend
Implement eviction as a subclass of the HuggingFace `Cache` API (`transformers`), one class per method, behind a common interface. This keeps the model code untouched and makes methods swappable.

```
src/caches/  base.py, full.py, random.py, recency.py,
             streaming.py, h2o.py, snapkv.py, pyramid.py
```

**FR1.1** Every cache exposes `current_bytes()` — the harness enforces budget in bytes, not tokens.
**FR1.2** Every cache logs per-layer token counts per step (needed to verify PyramidKV allocation and to catch silent budget violations).

### FR2 — Budget enforcement
A method must never exceed its byte budget. The harness **asserts** this every decode step and fails loudly. (Silent budget violation would invalidate the entire comparison — this is the single most important correctness property in the project.)

### FR3 — Prompt builder with position control
Construct each eval prompt in both **question-first** and **question-last** form from the same source document, with identical token counts where possible. Log both.

### FR4 — Measurement harness
- Wall-clock timing with **explicit device synchronization** (`torch.mps.synchronize()` / `torch.cuda.synchronize()`) inside every timed region — async dispatch otherwise measures queue submission, not execution
- Peak memory via `torch.mps.current_allocated_memory()` / `torch.cuda.max_memory_allocated()`, plus process RSS as a cross-check
- Cache clearing / warm-up between trials

### FR5 — Experiment runner
Config-driven (YAML), resumable, writes one JSON per cell to `results/`. Must survive interruption without losing completed cells.

### FR6 — Analysis
Scripts that read `results/` and emit the paper's tables and figures directly. No manual transcription of numbers into LaTeX — that is where errors enter.

---

## 6. Non-functional requirements

- **NFR1 Reproducibility.** One command reruns any cell. Seeds, model revisions (pinned commit hashes), and library versions recorded in every result file.
- **NFR2 Correctness over speed.** A wrong number is worse than a slow one.
- **NFR3 Portability.** Must run on Apple Silicon (MPS) *and* CUDA. See §7.
- **NFR4 Honest failure.** If a method cannot run in a configuration, record it as such — never silently substitute or interpolate.

---

## 7. Compute budget & feasibility

### 7.1 The binding constraint
KV cache size for MHA at 32k context, 7B model:
`2 (K,V) × 32 layers × 32 heads × 128 dim × 32768 tokens × 2 bytes ≈ 17 GB`

That does **not** fit alongside weights on a 16–32 GB Mac. For GQA (8 KV heads) the same figure is ≈ 4.3 GB, which does.

**Implication:** 32k on an MHA model is out of reach locally. Plan accordingly:
- Broad sweep on **small models** (1B–3B: Llama-3.2-1B/3B, Qwen2.5-1.5B/3B) — this is where most cells run
- Confirmation at **7–8B** on a narrower grid
- 32k context only on GQA models

### 7.2 Rent a GPU for the latency numbers
Do accuracy experiments locally. But publishing latency/throughput measured **only** on Apple Silicon MPS is a reviewer target — the same criticism that applies to the RMI project's GPU section.

Rent a single A10/L4 (~$0.50–1.00/hr) for ~20–40 hours to produce systems numbers on standard hardware. **Total cost ≈ $50.** Worth it; it removes an entire category of objection.

### 7.3 Risk: MPS operator gaps
`transformers` on MPS has op coverage gaps and silent CPU fallbacks that destroy timing validity. **Mitigation:** verify device placement of every tensor in the hot path early (week 2), before building on top of it. If MPS proves unworkable, accuracy work moves to CPU (slow but valid) and systems work to the rented GPU.

---

## 8. Timeline — planned backward from the application deadline

### 8.1 The binding date

The deadline that matters is **not** the venue's. No venue will have made a decision by
application time regardless, so what gets listed on the application is the **arXiv preprint**
plus "under review at X".

| Milestone | Date |
|---|---|
| UIUC / GaTech / UT Austin PhD applications | typically **Dec 15, 2026** (some Dec 1 — verify per school) |
| **⇒ HARD STOP: arXiv preprint live** | **Nov 20, 2026** |
| ⇒ venue submission | Oct–Nov, parallel to the above — not a gate |

arXiv moderation takes 1–2 days. Nov 20 leaves real buffer before a Dec 1 school.

All dates below assume a start of **Aug 6, 2026**. Verify every venue deadline directly;
those listed are typical patterns, not confirmed.

### 8.2 Venue assessment

| Venue | Typical deadline | Verdict |
|---|---|---|
| ICLR 2027 | ~late Sept 2026 | ❌ **Written off.** ~6 weeks out vs a 14-week plan. Do not half-attempt |
| NeurIPS 2026 workshops | ~Sept–Oct 2026 | ⚠️ Only viable as a 4–8pp workshop paper; decide by week 6 |
| **MLSys 2027** | ~Oct–Nov 2026 | ✅ **Primary target** |
| ICML 2027 | ~late Jan 2027 | ⏰ After applications; good follow-up with the extended version |
| arXiv | anytime | ✅ Regardless of the above |

### 8.3 Schedule

| Weeks | Dates | Milestone | Exit criterion |
|---|---|---|---|
| 1–2 | Aug 6–20 | Prior-art search; **reproduce ONE method end-to-end** (SnapKV) | Within a few % of published, or documented reason why not |
| 3–5 | Aug 20–Sep 10 | Cache API, methods, budget assertion, harness | Budget assertion passes for every method at every budget |
| 6–8 | Sep 10–Oct 1 | Main sweep (RQ1, RQ3) | Full grid: matched budgets + trivial baselines |
| 9–10 | Oct 1–15 | **Question-position experiment (RQ2)** ⭐ | H2 tested at all budgets on ≥2 models |
| 11–12 | Oct 15–29 | GQA transfer (RQ4); rented-GPU systems numbers | MHA vs GQA ordering table |
| 13–14 | Oct 29–Nov 12 | Writing | Complete draft |
| 15 | Nov 12–20 | Polish; **post arXiv** | 🔴 **HARD STOP — preprint live Nov 20** |
| — | Nov–Dec | Venue submission; applications | — |

### 8.4 Gates

**Week 2 — reproduction gate.** If one method cannot be reproduced end-to-end in two weeks,
the scope is wrong. Cut methods (drop to 4: full, random, recency, SnapKV) rather than slip.
The RQ2 result with 4 methods beats a broken sweep with 7.

**Week 10 — minimum-viable-paper gate.** By Oct 15 there must be enough for a complete paper
*even if everything after is cut*. Concretely: RQ1 + RQ2 + RQ3 on ≥2 models, written up.
RQ4 and the rented-GPU systems numbers are **explicitly droppable** — they strengthen the
paper but are not load-bearing.

If week 10 arrives without a viable paper, stop adding experiments and start writing.

### 8.5 What slips first, in order

When time runs short, cut in this sequence. Decide in advance so the choice isn't made under
pressure:

1. RQ4 (GQA transfer) — drop entirely
2. Rented-GPU systems numbers — report accuracy only, state the limitation
3. Method count 7 → 4
4. Context lengths 32k and 16k
5. Model count → 1

Never cut: the budget assertion, trivial baselines (RQ3), or the question-position
manipulation (RQ2). Those three *are* the paper.

---

## 9. Deliverables

1. `kv-bench` — open-source harness, reproducible, documented
2. Paper (8–9 pages) — target below
3. arXiv preprint
4. Results artifact (all JSON, versioned)
5. Pre-registration commit of H1–H4, timestamped before the main sweep

---

## 10. Target venues

Verify all deadlines directly — these shift year to year and are from before May 2026.

| Venue | Typical deadline | Fit |
|---|---|---|
| **NeurIPS/ICLR efficiency workshops** | Sept–Oct | Best fit; realistic acceptance before applications |
| **MLSys** | ~Oct | Strong fit for the systems half |
| **ICLR main** | ~Sept | Ambitious; OpenReview makes the submission public and citable immediately |
| **arXiv** | anytime | Do this regardless |

---

## 11. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Cannot reproduce a published method | **High** | Week-2 gate; cut method count, document the failure (a reproduction failure is itself reportable) |
| RQ2 finds no effect | Medium | Null result still publishable; RQ1/RQ3 carry the paper |
| MPS op gaps / silent CPU fallback | Medium | Verify device placement week 2; fall back to rented GPU |
| Scope creep (adding methods) | **High** | Method list is frozen after week 3. New ideas go in `notes/future.md` |
| Budget assertion reveals a method cannot hit low budgets | Low | That is a finding — report it |

---

## 12. Open questions to resolve in week 1

1. Which LongBench subsets? (Pick 4–6 spanning QA / summarization / retrieval; justify the choice)
2. Exact model list and pinned revisions
3. Does an existing harness (e.g. KVPress, LongBench eval code) cover enough to fork rather than build? **Check before writing code** — forking a trusted harness is faster and more credible than a fresh implementation
4. Is there already a paper doing matched-budget comparison? **Search hard for this in week 1.** If one exists, pivot to RQ2 alone, which is narrower and likely still unclaimed

Question 4 is the existential one. Resolve it first — before any code.

---

## 13. References to obtain and read (week 1)

Verify these against the actual papers; details below are recalled and may be imprecise.

- Xiao et al., *Efficient Streaming Language Models with Attention Sinks* (StreamingLLM), 2023
- Zhang et al., *H2O: Heavy-Hitter Oracle for Efficient Generative Inference*, 2023
- Li et al., *SnapKV: LLM Knows What You are Looking for Before Generation*, 2024
- Cai et al., *PyramidKV: Dynamic KV Cache Compression via Pyramidal Information Funneling*, 2024
- Liu et al., *Scissorhands: Exploiting the Persistence of Importance Hypothesis*, 2023
- Ge et al., *Model Tells You What to Discard* (FastGen), 2023
- Oren et al., *Transformers are Multi-State RNNs* (TOVA), 2023
- Bai et al., *LongBench*, 2023
- Kamradt, *Needle In A Haystack*, 2023
