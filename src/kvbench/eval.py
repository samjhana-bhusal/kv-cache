"""Phase-3 accuracy runner: task accuracy at matched BYTE budgets (not matched token fractions).

For each method we calibrate the nominal compression_ratio that hits each target byte fraction,
then evaluate accuracy on the synthetic needle task using the kvpress generation pipeline (which
compresses the context during prefill and excludes the question from compression). Masking methods
that cannot reach a byte target are recorded as INFEASIBLE, never interpolated.

  python -m kvbench.eval --config configs/eval_pilot.yaml

Writes one JSON per (model, ctx, method, byte_fraction) cell to results/. Device-agnostic.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvbench import bytes as kb
from kvbench import device as kvdev
from kvbench.presses import (
    REGISTRY,
    build_press,
    calibrate_ratio_for_byte_fraction,
)
from kvbench.run import EAGER_METHODS
from kvbench.tasks import make_needle, score
from kvbench import ruler as ruler_task


def measure_fraction_fn(model, tokenizer, modules, context, device):
    """Return f(ratio)->realized_fraction by prefilling `context` under the press at that ratio."""
    in_dev = kvdev.input_device(model, device)
    ids = tokenizer(context, return_tensors="pt").input_ids.to(in_dev)
    with torch.no_grad():
        full = kb.measure(model(ids, use_cache=True).past_key_values)

    def f(ratio: float) -> float:
        if ratio <= 0:
            return 1.0
        press = build_press(method_name.value, ratio)
        if hasattr(press, "post_init_from_model"):
            try:
                press.post_init_from_model(model)
            except Exception:  # noqa: BLE001
                pass
        with torch.no_grad(), press(model):
            rep = kb.measure(model(ids, use_cache=True).past_key_values, module_list=modules)
        return rep.total_bytes / full.total_bytes if full.total_bytes else 1.0

    return f, full.total_bytes


class _Box:
    """Tiny mutable holder so the closure above can see the current method name."""

    value: str = ""


method_name = _Box()


def _pred_text(out) -> str:
    if isinstance(out, dict):
        if isinstance(out.get("answers"), list) and out["answers"]:
            return str(out["answers"][0])
        if "answer" in out:
            return str(out["answer"])
    return str(out)


def _eval_accuracy(pipe, tokenizer, ctx, meth, ratio, n_items, task, ruler_items=None) -> float:
    press = build_press(meth, ratio)
    if hasattr(press, "post_init_from_model"):
        try:
            press.post_init_from_model(pipe.model)
        except Exception:  # noqa: BLE001
            pass
    hits, n = 0.0, 0
    if task == "ruler":
        for item in ruler_items:
            out = pipe(item.context, question=item.question, answer_prefix=item.answer_prefix,
                       press=press, max_new_tokens=item.max_new_tokens)
            hits += ruler_task.score_item(_pred_text(out), item)
            n += 1
    else:  # synthetic needle-in-a-haystack
        for seed in range(n_items):
            depth = (seed + 1) / (n_items + 1)
            item = make_needle(tokenizer, ctx, seed=seed, depth=depth)
            out = pipe(item.context, question=item.question, press=press, max_new_tokens=16)
            hits += score(_pred_text(out), item.answer)
            n += 1
    return hits / n if n else 0.0


def run(model, tokenizer, modules, cfg, model_name, device, results_dir):
    """Fixed-nominal-ratio accuracy + realized bytes per cell.

    Records BOTH accuracy and realized byte fraction at each nominal ratio, so the analysis can plot
    accuracy against the field's axis (nominal ratio) AND the honest axis (realized bytes) from the
    same cells — that contrast is Fig 2. Simulation-only methods (AdaKV) appear at realized~1.0
    regardless of ratio, exposing them on the byte axis.
    """
    from kvpress import KVPressTextGenerationPipeline

    pipe = KVPressTextGenerationPipeline(model=model, tokenizer=tokenizer)
    n_items = cfg.get("n_items", 5)
    task = cfg.get("task", "niah")
    tag = "rul" if task == "ruler" else "acc"

    for ctx in cfg["context_lengths"]:
        # RULER items for this context length (loaded once, reused across method/ratio cells).
        ruler_items = ruler_task.load_ruler(ctx, n_items) if task == "ruler" else None
        # Calibration context: for RULER use a real item's context (right token distribution).
        calib = ruler_items[0].context if task == "ruler" else make_needle(tokenizer, ctx, seed=999, depth=0.5).context
        for meth in cfg["methods"]:
            method_name.value = meth
            frac_fn, full_bytes = measure_fraction_fn(model, tokenizer, modules, calib, device)
            for ratio in cfg["nominal_ratios"]:
                cid = f"{tag}__{model_name.replace('/', '_')}__ctx{ctx}__{meth}__r{ratio:.2f}"
                path = os.path.join(results_dir, cid + ".json")
                if os.path.exists(path):
                    print(f"  skip {cid}")
                    continue
                try:
                    realized_frac = frac_fn(ratio)
                    acc = _eval_accuracy(pipe, tokenizer, ctx, meth, ratio, n_items, task, ruler_items)
                    rec = {
                        "cell": cid, "kind": "accuracy", "task": task,
                        "method": meth, "model": model_name,
                        "context_len": ctx, "nominal_ratio": ratio,
                        "realized_fraction": realized_frac,
                        "realized_bytes": int(realized_frac * full_bytes),
                        "full_cache_bytes": full_bytes, "accuracy": acc, "n_items": n_items,
                        "query_aware": REGISTRY[meth].query_aware,
                        "taxonomy_prior": REGISTRY[meth].taxonomy_prior,
                        "device": device.type, "timestamp": time.time(),
                    }
                    with open(path, "w") as f:
                        json.dump(rec, f, indent=2)
                    print(f"  {cid}: realized={realized_frac:.2f} acc={acc:.2f}")
                except Exception as e:  # noqa: BLE001 — honest failure, never interpolate
                    with open(path, "w") as f:
                        json.dump({"cell": cid, "error": repr(e), "failed": True}, f, indent=2)
                    print(f"  {cid}: FAILED {e!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    os.makedirs(args.results, exist_ok=True)
    device = kvdev.pick_device(args.device)
    dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
    print(f"device: {device}")

    device_map = cfg.get("device_map")  # e.g. "auto" for Kaggle T4x2 / multi-GPU
    for model_name in cfg["models"]:
        needs_eager = any(m in EAGER_METHODS for m in cfg["methods"])
        attn = "eager" if needs_eager else "sdpa"
        print(f"loading {model_name} (attn={attn}, device_map={device_map}) ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = kvdev.load_model(model_name, dtype, attn, device, device_map)
        modules = kb.collect_attention_modules(model)
        run(model, tokenizer, modules, cfg, model_name, device, args.results)
        del model
        kvdev.empty_cache(device)

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
