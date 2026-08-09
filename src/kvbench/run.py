"""Phase-2 audit runner: measure realized KV bytes per (model, context_len, press, ratio).

No accuracy evaluation here — this is the cheap, decisive figure. For each cell we prefill a fixed
context, apply the press, and measure the realized bytes of the resulting cache against the
full-cache baseline. Writes one JSON per cell to results/; resumable (skips existing cells).

  python -m kvbench.run --config configs/pilot.yaml

Device-agnostic (MPS/CUDA/CPU) via kvbench.device.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvbench import bytes as kb
from kvbench import device as kvdev
from kvbench.presses import REGISTRY, build_press, classify_measured

# Attention-score methods need the materialized attention matrix, so they cannot use SDPA/Flash
# kernels — they require eager attention. That incompatibility is itself a systems finding
# (cf. KeyDiff, arXiv:2504.15364); we record it rather than hide it.
EAGER_METHODS = {"observed_attention"}


def cell_id(model: str, ctx: int, press: str, ratio: float) -> str:
    safe_model = model.replace("/", "_")
    return f"{safe_model}__ctx{ctx}__{press}__r{ratio:.3f}"


def clear_press_state(modules) -> None:
    """Drop any masked_key_indices a previous press left on attention modules.

    Simulation-only presses (AdaKV, etc.) stash mask index tensors on the module; if left in place,
    a later forward can apply a stale mask whose dimensions no longer match the cache — a source of
    out-of-bounds gathers (and CUDA launch failures on GPU). Reset before every forward.
    """
    for mod in modules:
        if getattr(mod, "masked_key_indices", None) is not None:
            mod.masked_key_indices = None


def full_cache_baseline(model, input_ids, modules, device) -> kb.ByteReport:
    clear_press_state(modules)
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    kvdev.synchronize(device)
    return kb.measure(out.past_key_values)


def run_cell(model, modules, input_ids, ctx, press_name, ratio, full, device) -> dict:
    spec = REGISTRY[press_name]
    press = build_press(press_name, ratio)
    if hasattr(press, "post_init_from_model"):
        try:
            press.post_init_from_model(model)
        except Exception:  # noqa: BLE001 — some presses init lazily via the hook
            pass

    clear_press_state(modules)
    kvdev.empty_cache(device)
    kvdev.synchronize(device)
    # Measure the peak-prefill transient: extra memory allocated DURING compression beyond the
    # cache that remains after. Ratio accounting is blind to this (e.g. KVComposePress' ~2x pass).
    kvdev.reset_peak_memory(device)
    baseline_alloc = kvdev.current_allocated_bytes(device) or 0
    with torch.no_grad(), press(model):
        out = model(input_ids, use_cache=True)
    kvdev.synchronize(device)
    peak_alloc = kvdev.peak_allocated_bytes(device) or 0
    steady_alloc = kvdev.current_allocated_bytes(device) or 0
    # transient = peak during prefill − steady state after (weights cancel out; >=0 by construction).
    peak_prefill_transient = max(0, peak_alloc - steady_alloc)

    rep = kb.measure(out.past_key_values, module_list=modules)
    measured_class = classify_measured(full.total_bytes, rep.total_bytes)

    return {
        "method": press_name,
        "model": model.name_or_path,
        "context_len": int(input_ids.shape[-1]),
        "requested_context_len": ctx,
        "nominal_ratio": ratio,
        "realized_bytes": rep.total_bytes,
        "payload_bytes": rep.payload_bytes,
        "meta_bytes": rep.meta_bytes,
        "meta_breakdown": rep.meta_breakdown,
        "full_cache_bytes": full.total_bytes,
        "realized_fraction": rep.total_bytes / full.total_bytes if full.total_bytes else None,
        # Decomposition for Fig 3 — the measurable parts: resident payload, method metadata, and the
        # transient allocated during compression that ratio accounting never sees.
        "byte_decomposition": {
            "payload": rep.payload_bytes,
            "meta": rep.meta_bytes,
            "peak_prefill_transient": int(peak_prefill_transient),
        },
        "peak_prefill_transient_bytes": int(peak_prefill_transient),
        "baseline_alloc_bytes": int(baseline_alloc),
        "taxonomy_prior": spec.taxonomy_prior,
        "taxonomy_measured": measured_class,
        "query_aware": spec.query_aware,
        "per_layer_tokens": rep.per_layer_tokens,
        "n_kv_heads": rep.n_kv_heads,
        "head_dim": rep.head_dim,
        "itemsize": rep.itemsize,
        "device": device.type,
        "torch_version": torch.__version__,
        "timestamp": time.time(),
    }


def make_context(tokenizer, ctx_len: int, device) -> torch.Tensor:
    """Deterministic filler context of exactly ctx_len tokens (audit needs length, not meaning)."""
    text = ("The quick brown fox jumps over the lazy dog. " * ((ctx_len // 8) + 2))
    ids = tokenizer(text, return_tensors="pt").input_ids[:, :ctx_len]
    return ids.to(device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--device", default=None, help="force mps/cuda/cpu")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    os.makedirs(args.results, exist_ok=True)
    device = kvdev.pick_device(args.device)
    print(f"device: {device}")

    dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32

    device_map = cfg.get("device_map")
    for model_name in cfg["models"]:
        # Load ONE model per model_name. If any method needs eager attention, use eager for all —
        # holding two sharded models at once under device_map=auto risks a CUDA OOM/launch failure.
        # Attention impl does not change realized byte measurements, so this is safe for the audit.
        attn_impl = "eager" if any(m in EAGER_METHODS for m in cfg["methods"]) else "sdpa"
        print(f"loading {model_name} (attn={attn_impl}, device_map={device_map}) ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = kvdev.load_model(model_name, dtype, attn_impl, device, device_map, cfg.get("quantize"))
        in_dev = kvdev.input_device(model, device)

        modules = kb.collect_attention_modules(model)
        for ctx in cfg["context_lengths"]:
            input_ids = make_context(tokenizer, ctx, in_dev)
            # Full-cache baseline once per context (identical across methods) — halves the number
            # of forwards vs computing it per cell, and runs with clean (unmasked) module state.
            full = full_cache_baseline(model, input_ids, modules, device)
            for press_name in cfg["methods"]:
                for ratio in cfg["ratios"]:
                    cid = cell_id(model_name, ctx, press_name, ratio)
                    path = os.path.join(args.results, cid + ".json")
                    if os.path.exists(path):
                        print(f"  skip {cid}")
                        continue
                    try:
                        rec = run_cell(model, modules, input_ids, ctx, press_name, ratio, full, device)
                        with open(path, "w") as f:
                            json.dump(rec, f, indent=2)
                        frac = rec["realized_fraction"]
                        print(f"  {cid}: realized={frac:.3f} of full  [{rec['taxonomy_measured']}]")
                    except Exception as e:  # noqa: BLE001 — record honest failure, never interpolate
                        fail = {"cell": cid, "error": repr(e), "failed": True}
                        with open(path, "w") as f:
                            json.dump(fail, f, indent=2)
                        print(f"  {cid}: FAILED {e!r}")

        del model
        kvdev.empty_cache(device)

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
