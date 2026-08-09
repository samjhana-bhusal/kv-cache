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


def input_device(model, fallback: torch.device) -> torch.device:
    """Device to place input_ids on. With device_map sharding, use the embedding's device;
    otherwise the single device the model lives on."""
    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            return emb.weight.device
    except Exception:  # noqa: BLE001
        pass
    return fallback


def _balanced_max_memory(headroom_gib: float = 5.0):
    """Per-GPU memory caps that leave headroom for activations.

    `device_map='auto'` greedily fills GPU 0, then spills — so a 7B (~14 GB) lands entirely on one
    16 GB T4 with no room for the attention forward, and OOMs. Capping each GPU below its capacity
    forces accelerate to spread the weights and leaves room for activations. GPU 0 gets the most
    headroom because inputs/embeddings/activations accumulate there.
    """
    n = torch.cuda.device_count()
    if n <= 1:
        return None
    mm = {}
    for i in range(n):
        cap = torch.cuda.get_device_properties(i).total_memory / 1024**3
        reserve = headroom_gib if i == 0 else 1.5
        mm[i] = f"{max(1.0, cap - reserve):.0f}GiB"
    return mm


def load_model(model_name, dtype, attn_impl, device, device_map=None, quantize=None):
    """Load a causal LM portably across single-device (.to) and multi-GPU (device_map=auto).

    Uses `dtype=` (not the deprecated `torch_dtype=`). When device_map is set (e.g. "auto" on a
    Kaggle T4x2), the model is sharded across GPUs with per-GPU memory caps that reserve activation
    headroom (see _balanced_max_memory), and we do NOT call .to(device). `quantize='4bit'` loads
    weights in 4-bit (bitsandbytes) so a 7-8B model fits a single 16 GB GPU — the KV cache stays in
    `dtype`, so byte accounting is unaffected.
    """
    from transformers import AutoModelForCausalLM

    kwargs = dict(dtype=dtype, attn_implementation=attn_impl)
    if quantize == "4bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type="nf4")
        device_map = device_map or "auto"

    if device_map:
        if torch.cuda.is_available():
            print(f"  CUDA devices visible: {torch.cuda.device_count()}")
        mm = _balanced_max_memory() if device_map == "auto" and quantize != "4bit" else None
        if mm:
            kwargs["max_memory"] = mm
            print(f"  balanced max_memory: {mm}")
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, **kwargs)
        return model.eval()
    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs).to(device).eval()
