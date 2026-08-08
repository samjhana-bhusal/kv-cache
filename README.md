# kv-bench

Matched-budget head-to-head evaluation of KV cache eviction methods for LLM inference.

**Status:** pre-implementation. Read [SRS.md](SRS.md) first.

## The question

Published KV cache eviction methods (StreamingLLM, H2O, SnapKV, PyramidKV) each report
wins — but under different budget definitions, models, tasks, and baselines. This project
runs them under one controlled protocol, with budget fixed in **bytes**, and asks whether
the reported ordering survives.

Central hypothesis (**RQ2**): query-aware methods that peek at an observation window at the
end of the prompt may be exploiting the fact that benchmarks put the question last. Move the
question to the front, and their advantage should shrink.

## Layout

```
SRS.md            full specification — start here
src/caches/       one Cache subclass per eviction method
src/harness/      experiment runner, measurement, budget enforcement
configs/          YAML experiment definitions
results/          one JSON per experiment cell (versioned)
notes/            lit review, decisions, future ideas
```

## Before writing any code

Work §12 of the SRS, in order. Question 4 is existential:

> Does a matched-budget comparison paper already exist?

If yes, the project pivots to RQ2 alone. Find out first.

Also check whether an existing harness (KVPress, LongBench eval code) can be forked rather
than rebuilt — forking something trusted is faster *and* more credible than a fresh
implementation.

## Two invariants

1. **Budget is enforced in bytes and asserted every decode step.** A silent budget violation
   invalidates every comparison in the paper.
2. **Every timed region has an explicit device sync.** MPS and CUDA dispatch asynchronously;
   without a sync you measure queue submission, not execution.
