"""Instrument self-tests — run before trusting any measurement.

  python -m kvbench.selftest

Checks (paper §Method sanity control):
  1. On a synthetic full cache with known dims, measure() reproduces the analytic byte figure exactly.
  2. Physically slicing K/V (memory-faithful path) reduces measured payload proportionally.
  3. Masking without slicing (simulation-only path) leaves payload unchanged — the positive control.
"""
from __future__ import annotations

import sys

import torch
from transformers import DynamicCache

from kvbench import bytes as kb


def _make_cache(n_layers, batch, n_kv_heads, seq, head_dim, dtype=torch.float32) -> DynamicCache:
    c = DynamicCache()
    for layer in range(n_layers):
        k = torch.zeros(batch, n_kv_heads, seq, head_dim, dtype=dtype)
        v = torch.zeros(batch, n_kv_heads, seq, head_dim, dtype=dtype)
        c.update(k, v, layer)
    return c


def main() -> int:
    n_layers, batch, n_kv_heads, seq, head_dim = 4, 1, 8, 128, 64
    itemsize = torch.finfo(torch.float32).bits // 8

    # 1. analytic reproduction
    c = _make_cache(n_layers, batch, n_kv_heads, seq, head_dim)
    rep = kb.measure(c)
    analytic = kb.analytic_full_cache_bytes(n_layers, n_kv_heads, head_dim, seq, itemsize, batch)
    ok1 = rep.payload_bytes == analytic
    print(f"[1] payload={rep.payload_bytes} analytic={analytic}  {'OK' if ok1 else 'MISMATCH'}")

    # 2. memory-faithful: physically gather to half the tokens
    kept = seq // 2
    for layer in c.layers:
        idx = torch.arange(kept).view(1, 1, kept, 1).expand(batch, n_kv_heads, kept, head_dim)
        layer.keys = layer.keys.gather(2, idx).contiguous()
        layer.values = layer.values.gather(2, idx).contiguous()
    rep2 = kb.measure(c)
    ok2 = rep2.payload_bytes == analytic // 2
    print(f"[2] sliced payload={rep2.payload_bytes} expected={analytic // 2}  "
          f"{'OK' if ok2 else 'MISMATCH'}")

    # 3. simulation-only positive control: full cache + a mask index tensor => payload unchanged,
    #    meta > 0. This is exactly AdaKVPress's behaviour.
    c3 = _make_cache(n_layers, batch, n_kv_heads, seq, head_dim)

    class _FakeAttn:
        pass

    mod = _FakeAttn()
    n_pruned = n_kv_heads * (seq - kept)
    mod.masked_key_indices = (
        torch.arange(batch).repeat_interleave(n_pruned),
        torch.zeros(n_pruned, dtype=torch.long),
        torch.zeros(n_pruned, dtype=torch.long),
    )
    rep3 = kb.measure(c3, module_list=[mod])
    ok3 = rep3.payload_bytes == analytic and rep3.meta_bytes > 0
    print(f"[3] masked payload={rep3.payload_bytes} (== full {analytic}), "
          f"meta={rep3.meta_bytes}  {'OK' if ok3 else 'MISMATCH'}")
    print("    ^ positive control: a masking method reports compression but frees ZERO payload bytes.")

    passed = ok1 and ok2 and ok3
    print(f"\n{'ALL PASS' if passed else 'FAILURES PRESENT'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
