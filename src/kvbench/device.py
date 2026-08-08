"""Device selection and synchronization — device-agnostic across MPS / CUDA / CPU.

SRS FR4: every timed region must contain an explicit device sync. MPS and CUDA dispatch
asynchronously; without a sync you measure queue-submission time, not execution. This module is
the single place that knows which backend is live, so timing and memory code stay portable between
the local Mac (MPS) and the later Ubuntu box (CUDA).
"""
from __future__ import annotations

import torch


def pick_device(prefer: str | None = None) -> torch.device:
    """Return the best available device, or the requested one if available."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    """Block until all queued work on `device` has executed. No-op on CPU."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    # MPS has no peak-reset; we sample current_allocated_memory around regions instead.


def peak_allocated_bytes(device: torch.device) -> int | None:
    """Peak allocator bytes since last reset, or None if the backend can't report it."""
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated())
    if device.type == "mps":
        # MPS exposes current, not peak; callers sample this at the region's high-water point.
        return int(torch.mps.current_allocated_memory())
    return None


def current_allocated_bytes(device: torch.device) -> int | None:
    if device.type == "cuda":
        return int(torch.cuda.memory_allocated())
    if device.type == "mps":
        return int(torch.mps.current_allocated_memory())
    return None


def empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def load_model(model_name, dtype, attn_impl, device, device_map=None):
    """Load a causal LM portably across single-device (.to) and multi-GPU (device_map=auto).

    Uses `dtype=` (not the deprecated `torch_dtype=`). When device_map is set (e.g. "auto" on a
    Kaggle T4x2), transformers shards the model across GPUs and we do NOT call .to(device).
    """
    from transformers import AutoModelForCausalLM

    kwargs = dict(dtype=dtype, attn_implementation=attn_impl)
    if device_map:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, **kwargs)
        return model.eval()
    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs).to(device).eval()
