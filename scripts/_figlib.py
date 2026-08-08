"""Shared helpers for figure scripts.

Enforces the project's non-negotiable: a figure renders from real result files or it does not
render. `load_cells` raises (non-zero exit) if the results directory is missing or empty, so the
Makefile's `make paper` fails loudly rather than emitting a plot built on nothing.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def standard_args(description: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--results", required=True, help="directory of results/*.json cells")
    ap.add_argument("--out", required=True, help="output PDF path")
    return ap.parse_args()


def load_cells(results_dir: str) -> list[dict]:
    """Load every results/*.json cell. Exit non-zero if none exist — never fabricate."""
    if not os.path.isdir(results_dir):
        sys.exit(f"[figlib] results dir '{results_dir}' does not exist — run experiments first.")
    paths = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    if not paths:
        sys.exit(
            f"[figlib] no result cells in '{results_dir}'. "
            "This figure has no data yet; refusing to emit a placeholder plot. "
            "Run the sweep (src/kvbench/run.py) to populate results/."
        )
    cells = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            cells.append(json.load(f))
    return cells


# Expected schema of a results/*.json cell (documented here so figure scripts and the runner agree).
# Populated by src/kvbench/run.py once the harness exists:
#   {
#     "method": "snapkv",            # press name
#     "model": "Qwen2.5-7B-Instruct",
#     "context_len": 8192,
#     "nominal_ratio": 0.5,          # the field's knob
#     "realized_bytes": 1234567,     # measured via bytes.py Eq.(1)
#     "full_cache_bytes": 2345678,   # measured full-cache baseline for the same cell
#     "peak_prefill_bytes": 3456789,
#     "taxonomy_class": "simulation-only" | "memory-faithful" | "incommensurable",
#     "byte_budget": 1048576,        # target for byte-matched cells (else null)
#     "accuracy": 0.63,              # task metric
#     "decode_tokens_per_s": 42.0,
#     "seed": 0,
#     "byte_decomposition": {"payload":..., "sinks":..., "recency":..., "obs_window":..., "meta":...}
#   }
