#!/usr/bin/env python3
"""Fig 3 — per-method byte decomposition (stacked bar), at a fixed nominal ratio.

Shows WHERE each method's bytes are, using the three measurable components the audit records:
resident K/V payload, method metadata (mask indices / accumulators), and the peak-prefill transient
(extra memory allocated during compression beyond the steady-state cache). The transient is the
component ratio accounting is completely blind to. Renders only from real audit results/*.json that
carry a byte_decomposition field.
"""
from __future__ import annotations

import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _figlib import load_cells, standard_args  # noqa: E402

COMPONENTS = ["payload", "meta", "peak_prefill_transient"]
LABELS = {"payload": "K/V payload", "meta": "metadata", "peak_prefill_transient": "peak-prefill transient"}


def main() -> None:
    args = standard_args(__doc__)
    cells = [c for c in load_cells(args.results)
             if "byte_decomposition" in c and not c.get("failed") and c.get("kind") != "accuracy"]
    if not cells:
        sys.exit("[fig3] no audit cells with byte_decomposition — run kvbench.run (updated) first.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Single model, single context (largest), nearest ratio to 0.5 per method.
    model = max({c["model"] for c in cells}, key=lambda m: sum(c["model"] == m for c in cells))
    cells = [c for c in cells if c["model"] == model]
    ctx = max(c["context_len"] for c in cells)
    cells = [c for c in cells if c["context_len"] == ctx]

    best: dict[str, dict] = {}
    for c in cells:
        m = c["method"]
        if m not in best or abs(c["nominal_ratio"] - 0.5) < abs(best[m]["nominal_ratio"] - 0.5):
            best[m] = c

    methods = sorted(best)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    bottoms = np.zeros(len(methods))
    for comp in COMPONENTS:
        vals = np.array([best[m]["byte_decomposition"].get(comp, 0) / 1e6 for m in methods], float)
        ax.bar(methods, vals, bottom=bottoms, label=LABELS[comp])
        bottoms += vals
    full_mb = best[methods[0]].get("full_cache_bytes", 0) / 1e6
    if full_mb:
        ax.axhline(full_mb, color="0.4", ls="--", lw=1, label="full-cache payload")
    ax.set_ylabel("bytes (MB)")
    ax.set_title(f"Byte decomposition at nominal ratio ~0.5\n{model.split('/')[-1]}, ctx={ctx}")
    ax.legend(fontsize=8)
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"[fig3] wrote {args.out} for {len(methods)} methods ({model}, ctx={ctx})")


if __name__ == "__main__":
    main()
