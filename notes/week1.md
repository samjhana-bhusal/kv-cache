# Week 1 checklist

Do these in order. Do not write harness code until item 4 is answered.

## 1. Prior-art search (existential — do first)

Search hard for an existing matched-budget comparison of KV eviction methods.

Queries to run on arXiv / Semantic Scholar / Google Scholar:
- "KV cache compression survey"
- "KV cache eviction comparison"
- "fair comparison KV cache"
- "revisiting KV cache compression"
- "KV cache benchmark matched budget"
- Check citations *of* SnapKV and PyramidKV — a critique paper would cite both
- Check for recent "reality check" / "are we making progress" style papers in efficient inference

**Outcome:**
- [ ] No such paper → proceed with full SRS
- [ ] Partial overlap → narrow to RQ2 (question-position confound), which is likely still unclaimed
- [ ] Direct hit → reassess; RQ2 may still stand alone

Record findings in `notes/prior-art.md` with links.

## 2. Harness decision

Check whether these cover enough to fork:
- [ ] KVPress (NVIDIA) — has several methods implemented
- [ ] LongBench official eval code
- [ ] Any method's own repo that already implements competitors

Forking a trusted harness is **faster and more credible** than a fresh implementation.
Only build from scratch if nothing fits.

## 3. Environment sanity

- [ ] `transformers` + MPS: load a 1B model, generate, confirm output is coherent
- [ ] Verify every hot-path tensor is actually on MPS — check for silent CPU fallback
- [ ] Confirm `torch.mps.current_allocated_memory()` tracks as expected
- [ ] Time a generation with and without `torch.mps.synchronize()` — confirm they differ
      (if they don't, something is already forcing a sync; find out what)

## 4. Pick the eval set

- [ ] 4–6 LongBench subsets spanning QA / summarization / retrieval
- [ ] Justify each choice in writing (reviewers ask)
- [ ] Confirm each subset's prompts can be restructured question-first without breaking the task
      — **this gates RQ2**, so check it before committing to a subset

## 5. Pin models

- [ ] One MHA model (Llama-2-7B class) — for comparability with prior work
- [ ] One GQA model (Llama-3.x / Qwen2.5 class) — for RQ4
- [ ] One small model (1–3B) for the broad sweep
- [ ] Record exact HF revision hashes

## 6. Reproduce ONE method

Target: SnapKV, end-to-end, on one LongBench subset.

- [ ] Numbers within a few % of the published result
- [ ] OR a written explanation of the discrepancy

**This is the week-2 gate.** If it slips, cut method count — do not slip the schedule.
