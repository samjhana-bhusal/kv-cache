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


def full_cache_baseline(model, tokenizer, input_ids, device) -> kb.ByteReport:
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    return kb.measure(out.past_key_values)


def run_cell(model, tokenizer, modules, input_ids, ctx, press_name, ratio, device) -> dict:
    spec = REGISTRY[press_name]
    press = build_press(press_name, ratio)
    if hasattr(press, "post_init_from_model"):
        try:
            press.post_init_from_model(model)
        except Exception:  # noqa: BLE001 — some presses init lazily via the hook
            pass

    kvdev.empty_cache(device)
    kvdev.synchronize(device)
    with torch.no_grad(), press(model):
        out = model(input_ids, use_cache=True)
    kvdev.synchronize(device)

    rep = kb.measure(out.past_key_values, module_list=modules)
    full = full_cache_baseline(model, tokenizer, input_ids, device)
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

    for model_name in cfg["models"]:
        print(f"loading {model_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        models: dict[str, object] = {}  # attn_impl -> model, loaded lazily

        device_map = cfg.get("device_map")

        def get_model(attn_impl: str):
            if attn_impl not in models:
                models[attn_impl] = kvdev.load_model(
                    model_name, dtype, attn_impl, device, device_map
                )
            return models[attn_impl]

        for ctx in cfg["context_lengths"]:
            for press_name in cfg["methods"]:
                attn_impl = "eager" if press_name in EAGER_METHODS else "sdpa"
                model = get_model(attn_impl)
                modules = kb.collect_attention_modules(model)
                input_ids = make_context(tokenizer, ctx, device)
                for ratio in cfg["ratios"]:
                    cid = cell_id(model_name, ctx, press_name, ratio)
                    path = os.path.join(args.results, cid + ".json")
                    if os.path.exists(path):
                        print(f"  skip {cid}")
                        continue
                    try:
                        rec = run_cell(model, tokenizer, modules, input_ids, ctx, press_name, ratio, device)
                        with open(path, "w") as f:
                            json.dump(rec, f, indent=2)
                        frac = rec["realized_fraction"]
                        print(f"  {cid}: realized={frac:.3f} of full  [{rec['taxonomy_measured']}]")
                    except Exception as e:  # noqa: BLE001 — record honest failure, never interpolate
                        fail = {"cell": cid, "error": repr(e), "failed": True}
                        with open(path, "w") as f:
                            json.dump(fail, f, indent=2)
                        print(f"  {cid}: FAILED {e!r}")

        models.clear()
        kvdev.empty_cache(device)

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
