"""Press registry, memory-faithfulness taxonomy, and the byte-matched budget wrapper.

The taxonomy is the paper's organizing contribution. We assign a *prior* class by reading the
kvpress source (which compression path each press takes), and the audit runner *confirms* it
dynamically by measuring realized bytes. Prior and measurement must agree; a disagreement is a
finding.
"""
from __future__ import annotations

from dataclasses import dataclass

from kvpress import (
    AdaKVPress,
    ExpectedAttentionPress,
    KnormPress,
    ObservedAttentionPress,
    RandomPress,
    SnapKVPress,
    StreamingLLMPress,
    ThinKPress,
    TOVAPress,
)

# Taxonomy prior, from source inspection of kvpress 0.5.4:
#   memory-faithful  : ScorerPress.compress gathers K/V to n_kept and returns the smaller tensors.
#   simulation-only  : compress sets module.masked_key_indices and returns K/V UNMODIFIED; the
#                      attention_patch masks at compute time. Payload never shrinks.
#   incommensurable  : compresses off the token axis (ThinK prunes head_dim channels), so a
#                      token-fraction ratio is not a statement about token count at all.
MEMORY_FAITHFUL = "memory-faithful"
SIMULATION_ONLY = "simulation-only"
INCOMMENSURABLE = "incommensurable"


@dataclass
class PressSpec:
    name: str
    taxonomy_prior: str
    query_aware: bool  # RQ2 grouping — does scoring peek at an end-of-prompt observation window?
    build: callable    # (compression_ratio) -> press


def _snapkv(r):
    return SnapKVPress(compression_ratio=r)


def _adakv(r):
    # AdaKV wraps a ScorerPress; SnapKV is the canonical inner scorer. AdaKV itself only masks.
    return AdaKVPress(press=SnapKVPress(compression_ratio=r))


REGISTRY: dict[str, PressSpec] = {
    "random": PressSpec("random", MEMORY_FAITHFUL, False, lambda r: RandomPress(compression_ratio=r)),
    "knorm": PressSpec("knorm", MEMORY_FAITHFUL, False, lambda r: KnormPress(compression_ratio=r)),
    "streaming_llm": PressSpec(
        "streaming_llm", MEMORY_FAITHFUL, False, lambda r: StreamingLLMPress(compression_ratio=r)
    ),
    "tova": PressSpec("tova", MEMORY_FAITHFUL, False, lambda r: TOVAPress(compression_ratio=r)),
    "snapkv": PressSpec("snapkv", MEMORY_FAITHFUL, True, _snapkv),
    "expected_attention": PressSpec(
        "expected_attention", MEMORY_FAITHFUL, False,
        lambda r: ExpectedAttentionPress(compression_ratio=r),
    ),
    "observed_attention": PressSpec(
        "observed_attention", MEMORY_FAITHFUL, False,
        lambda r: ObservedAttentionPress(compression_ratio=r),
    ),  # closest thing kvpress ships to H2O (accumulated observed attention)
    "adakv": PressSpec("adakv", SIMULATION_ONLY, True, _adakv),
    "think": PressSpec("think", INCOMMENSURABLE, False, lambda r: ThinKPress(key_channel_compression_ratio=r)),
}


def build_press(name: str, ratio: float):
    if name not in REGISTRY:
        raise KeyError(f"unknown press '{name}'; known: {sorted(REGISTRY)}")
    return REGISTRY[name].build(ratio)


def classify_measured(full_cache_bytes: int, realized_bytes: int, tol: float = 0.02) -> str:
    """Dynamic taxonomy from measurement: near-full payload at high ratio => simulation-only."""
    frac = realized_bytes / full_cache_bytes if full_cache_bytes else 1.0
    if frac >= 1.0 - tol:
        return SIMULATION_ONLY
    return MEMORY_FAITHFUL  # incommensurable is decided by construction (ThinK), not by this test
