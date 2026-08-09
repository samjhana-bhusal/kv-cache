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
from kvbench import longbench as lb_task


def measure_fraction_fn(model, tokenizer, modules, context, device, max_ctx=None):
    """Return f(ratio)->realized_fraction by prefilling `context` under the press at that ratio.

    `max_ctx` truncates the calibration context — essential when the calibration item is a raw
    LongBench context (30k+ tokens), whose full-attention forward would OOM. RULER contexts are
    already fixed-length so this is a no-op for them."""
    in_dev = kvdev.input_device(model, device)
    ids = tokenizer(context, return_tensors="pt").input_ids
    if max_ctx:
        ids = ids[:, :max_ctx]
    ids = ids.to(in_dev)
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


def _time_call(pipe, device, context, **kwargs) -> tuple[str, float, int]:
    """Run one pipeline call with device sync around it; return (text, seconds, new_tokens).
    `context` is passed positionally — the kvpress pipeline requires it as the first argument."""
    kwargs = {k: v for k, v in kwargs.items() if v is not None}  # drop unset (e.g. max_context_length)
    kvdev.synchronize(device)
    t0 = time.perf_counter()
    out = pipe(context, **kwargs)
    kvdev.synchronize(device)
    dt = time.perf_counter() - t0
    text = _pred_text(out)
    new_tok = len(pipe.tokenizer(text).input_ids) if text else 0
    return text, dt, new_tok


def _eval_accuracy(pipe, tokenizer, ctx, meth, ratio, n_items, task, device, ruler_items=None,
                   longbench_items=None, max_ctx=None):
    """Return (accuracy, generation_tokens_per_s). Throughput is device-synced end-to-end
    (prefill+decode) generation rate, aggregated over the cell's items. `max_ctx` caps the context
    fed to the pipeline (essential for LongBench, whose natural contexts reach 30k+ tokens and would
    OOM a T4); it also makes LongBench context-comparable to RULER at the same budget."""
    press = build_press(meth, ratio)
    if hasattr(press, "post_init_from_model"):
        try:
            press.post_init_from_model(pipe.model)
        except Exception:  # noqa: BLE001
            pass
    hits, n = 0.0, 0
    tot_time, tot_tok = 0.0, 0
    if task == "ruler":
        for item in ruler_items:
            text, dt, ntok = _time_call(pipe, device, item.context, question=item.question,
                                        answer_prefix=item.answer_prefix, press=press,
                                        max_new_tokens=item.max_new_tokens, max_context_length=max_ctx)
            hits += ruler_task.score_item(text, item); n += 1
            tot_time += dt; tot_tok += ntok
    elif task == "longbench":
        for item in longbench_items:
            text, dt, ntok = _time_call(pipe, device, item.context, question=item.question,
                                        answer_prefix=item.answer_prefix, press=press,
                                        max_new_tokens=item.max_new_tokens, max_context_length=max_ctx)
            hits += lb_task.score_item(text, item); n += 1
            tot_time += dt; tot_tok += ntok
    else:  # synthetic needle-in-a-haystack
        for seed in range(n_items):
            depth = (seed + 1) / (n_items + 1)
            item = make_needle(tokenizer, ctx, seed=seed, depth=depth)
            text, dt, ntok = _time_call(pipe, device, item.context, question=item.question,
                                        press=press, max_new_tokens=16)
            hits += score(text, item.answer); n += 1
            tot_time += dt; tot_tok += ntok
    acc = hits / n if n else 0.0
    tps = tot_tok / tot_time if tot_time > 0 else None
    return acc, tps


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
    tag = {"ruler": "rul", "longbench": "lbn"}.get(task, "acc")

    for ctx in cfg["context_lengths"]:
        max_ctx = cfg.get("max_context_length", ctx)  # cap for LongBench's 30k+ contexts
        # Load benchmark items for this context length once, reuse across method/ratio cells.
        ruler_items = ruler_task.load_ruler(ctx, n_items) if task == "ruler" else None
        longbench_items = lb_task.load_longbench(n_items) if task == "longbench" else None
        # Calibration context: use a real item's context so the token distribution matches.
        if task == "ruler":
            calib = ruler_items[0].context
        elif task == "longbench":
            calib = longbench_items[0].context
        else:
            calib = make_needle(tokenizer, ctx, seed=999, depth=0.5).context
        for meth in cfg["methods"]:
            method_name.value = meth
            frac_fn, full_bytes = measure_fraction_fn(model, tokenizer, modules, calib, device,
                                                      max_ctx=max_ctx)
            for ratio in cfg["nominal_ratios"]:
                cid = f"{tag}__{model_name.replace('/', '_')}__ctx{ctx}__{meth}__r{ratio:.2f}"
                path = os.path.join(results_dir, cid + ".json")
                if os.path.exists(path):
                    print(f"  skip {cid}")
                    continue
                try:
                    realized_frac = frac_fn(ratio)
                    # Cap context to this cell's budget — critical for LongBench (30k+ contexts).
                    max_ctx = cfg.get("max_context_length", ctx)
                    acc, tps = _eval_accuracy(pipe, tokenizer, ctx, meth, ratio, n_items, task,
                                              device, ruler_items, longbench_items, max_ctx=max_ctx)
                    rec = {
                        "cell": cid, "kind": "accuracy", "task": task,
                        "method": meth, "model": model_name,
                        "context_len": ctx, "nominal_ratio": ratio,
                        "realized_fraction": realized_frac,
                        "realized_bytes": int(realized_frac * full_bytes),
                        "full_cache_bytes": full_bytes, "accuracy": acc, "n_items": n_items,
                        "decode_tokens_per_s": tps,
                        "query_aware": REGISTRY[meth].query_aware,
                        "taxonomy_prior": REGISTRY[meth].taxonomy_prior,
                        "device": device.type, "timestamp": time.time(),
                    }
                    with open(path, "w") as f:
                        json.dump(rec, f, indent=2)
                    tps_s = f"{tps:.1f} tok/s" if tps else "n/a"
                    print(f"  {cid}: realized={realized_frac:.2f} acc={acc:.2f} {tps_s}")
                except Exception as e:  # noqa: BLE001 — honest failure, never interpolate
                    with open(path, "w") as f:
                        json.dump({"cell": cid, "error": repr(e), "failed": True}, f, indent=2)
                    print(f"  {cid}: FAILED {e!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--device", default=None)
    ap.add_argument("--task", default=None, help="override cfg task: niah|ruler|longbench")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.task:
        cfg["task"] = args.task  # one config can drive multiple benchmarks across notebook cells
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
        model = kvdev.load_model(model_name, dtype, attn, device, device_map, cfg.get("quantize"))
        modules = kb.collect_attention_modules(model)
        run(model, tokenizer, modules, cfg, model_name, device, args.results)
        del model
        kvdev.empty_cache(device)

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
