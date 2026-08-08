"""The measurement instrument: realized bytes of a KV cache, read from stored tensors.

This is the paper's contribution in code. Every other benchmark infers memory from a nominal token
count; we measure the tensors that actually exist. A method that masks without slicing has
full-size key/value tensors and therefore full payload, regardless of the compression ratio it
reports — which is exactly how this instrument exposes simulation-only compression.

Realized bytes (paper Eq. 1):

    bytes(m) = sum_layers (|K_l| + |V_l|) * itemsize     # payload, from the STORED tensor
             + meta(m)                                    # accumulators, masks, tables

Sanity control: for a full cache this must reproduce the analytic
    2 * n_layers * n_kv_heads * head_dim * n_tokens * itemsize.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class ByteReport:
    payload_bytes: int
    meta_bytes: int
    n_layers: int
    n_kv_heads: int
    head_dim: int
    seq_len: int
    itemsize: int
    per_layer_tokens: list[int] = field(default_factory=list)
    meta_breakdown: dict[str, int] = field(default_factory=dict)

    @property
    def total_bytes(self) -> int:
        return self.payload_bytes + self.meta_bytes

    def as_dict(self) -> dict:
        return {
            "payload_bytes": self.payload_bytes,
            "meta_bytes": self.meta_bytes,
            "total_bytes": self.total_bytes,
            "n_layers": self.n_layers,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "seq_len": self.seq_len,
            "itemsize": self.itemsize,
            "per_layer_tokens": self.per_layer_tokens,
            "meta_breakdown": self.meta_breakdown,
        }


def _iter_layers(cache):
    """Yield (keys, values) per layer across the transformers>=5 DynamicCache API and older ones.

    transformers>=5: cache.layers[i].keys / .values
    older:           cache.key_cache[i] / cache.value_cache[i]
    """
    layers = getattr(cache, "layers", None)
    if layers is not None:
        for layer in layers:
            k = getattr(layer, "keys", None)
            v = getattr(layer, "values", None)
            if k is None or v is None:
                continue
            yield k, v
        return
    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if key_cache is not None and value_cache is not None:
        for k, v in zip(key_cache, value_cache):
            if k is None or v is None:
                continue
            yield k, v
        return
    raise TypeError(f"Unrecognized cache type {type(cache).__name__}: no .layers or .key_cache")


def payload_bytes(cache) -> tuple[int, list[int], dict]:
    """Bytes actually occupied by K/V tensors, plus per-layer token counts and shape info."""
    total = 0
    per_layer_tokens: list[int] = []
    shape_info: dict = {}
    for k, v in _iter_layers(cache):
        # [batch, n_kv_heads, seq, head_dim]
        total += k.numel() * k.element_size()
        total += v.numel() * v.element_size()
        per_layer_tokens.append(int(k.shape[-2]))
        if not shape_info:
            shape_info = {
                "n_kv_heads": int(k.shape[1]),
                "head_dim": int(k.shape[-1]),
                "itemsize": int(k.element_size()),
                "batch": int(k.shape[0]),
            }
    return total, per_layer_tokens, shape_info


def meta_bytes(cache, module_list=None) -> tuple[int, dict]:
    """Bytes of per-method bookkeeping that a token-fraction budget ignores.

    Covers the state kvpress attaches to attention modules:
      - masked_key_indices  (simulation-only presses: AdaKV, kvzip, criticalkv, duo, ...)
      - any accumulator/score tensors a press stashes on the module

    A simulation-only press keeps full payload AND carries mask index tensors — both are real
    resident bytes that the reported compression ratio omits. We surface them rather than hide them.
    """
    breakdown: dict[str, int] = {}
    total = 0
    if module_list is None:
        return 0, breakdown
    for mod in module_list:
        mki = getattr(mod, "masked_key_indices", None)
        if mki is not None:
            b = 0
            seq = mki if isinstance(mki, (tuple, list)) else [mki]
            for t in seq:
                if isinstance(t, torch.Tensor):
                    b += t.numel() * t.element_size()
            breakdown["masked_key_indices"] = breakdown.get("masked_key_indices", 0) + b
            total += b
    return total, breakdown


def measure(cache, module_list=None) -> ByteReport:
    """Full realized-byte report for a cache (paper Eq. 1)."""
    payload, per_layer_tokens, shape = payload_bytes(cache)
    meta, meta_break = meta_bytes(cache, module_list)
    return ByteReport(
        payload_bytes=payload,
        meta_bytes=meta,
        n_layers=len(per_layer_tokens),
        n_kv_heads=shape.get("n_kv_heads", 0),
        head_dim=shape.get("head_dim", 0),
        seq_len=max(per_layer_tokens) if per_layer_tokens else 0,
        itemsize=shape.get("itemsize", 0),
        per_layer_tokens=per_layer_tokens,
        meta_breakdown=meta_break,
    )


def analytic_full_cache_bytes(
    n_layers: int, n_kv_heads: int, head_dim: int, n_tokens: int, itemsize: int, batch: int = 1
) -> int:
    """The full-cache byte figure a correct instrument must reproduce (sanity control)."""
    return 2 * n_layers * n_kv_heads * head_dim * n_tokens * itemsize * batch


def collect_attention_modules(model) -> list:
    """Attention submodules kvpress may attach state to (for meta-byte accounting)."""
    mods = []
    for _, module in model.named_modules():
        # kvpress attaches masked_key_indices to self_attn modules; match by attribute surface.
        if hasattr(module, "masked_key_indices") or module.__class__.__name__.endswith("Attention"):
            mods.append(module)
    return mods
