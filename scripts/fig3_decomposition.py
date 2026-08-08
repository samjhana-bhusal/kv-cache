#!/usr/bin/env python3
"""Fig 3 — per-method byte decomposition at a fixed nominal ratio (stacked bar).

Shows WHERE each method's bytes live: payload, sinks, recency window, observation window, metadata,
and the peak-prefill transient. Renders only from real results/*.json.
"""
from __future__ import annotations

import sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _figlib import load_cells, standard_args  # noqa: E402

COMPONENTS = ["payload", "sinks", "recency", "obs_window", "meta"]


def main() -> None:
    args = standard_args(__doc__)
    cells = load_cells(args.results)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # One bar per method; take the cell nearest ratio=0.5 for each.
    best: dict[str, dict] = {}
    for c in cells:
        if "byte_decomposition" not in c:
            continue
        m = c["method"]
        if m not in best or abs(c["nominal_ratio"] - 0.5) < abs(best[m]["nominal_ratio"] - 0.5):
            best[m] = c
    if not best:
        sys.exit("[fig3] no cells carried byte_decomposition — cannot plot.")

    methods = sorted(best)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bottoms = np.zeros(len(methods))
    for comp in COMPONENTS:
        vals = np.array([best[m]["byte_decomposition"].get(comp, 0) for m in methods], float)
        ax.bar(methods, vals, bottom=bottoms, label=comp)
        bottoms += vals
    ax.set_ylabel("bytes")
    ax.set_title("Byte decomposition at nominal ratio ~0.5")
    ax.legend(fontsize=8)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"[fig3] wrote {args.out} for {len(methods)} methods")


if __name__ == "__main__":
    main()
