#!/usr/bin/env python3
"""Fig 5 — accuracy vs decode throughput at matched bytes (scatter, one point per method/budget).

Tests the systems half of the thesis: methods that free no bytes also buy no speedup, while being
ranked as if they saved both. Renders only from real results/*.json.
"""
from __future__ import annotations

import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _figlib import load_cells, standard_args, filter_model, savefig_both  # noqa: E402


def main() -> None:
    args = standard_args(__doc__)
    cells = filter_model(load_cells(args.results), args.model)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = [
        c
        for c in cells
        if c.get("accuracy") is not None and c.get("decode_tokens_per_s") is not None
    ]
    if not pts:
        sys.exit("[fig5] no cells carried both accuracy and decode_tokens_per_s — run 2b first.")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    methods = sorted({c["method"] for c in pts})
    cmap = plt.get_cmap("tab10")
    for i, m in enumerate(methods):
        mp = [c for c in pts if c["method"] == m]
        ax.scatter(
            [c["decode_tokens_per_s"] for c in mp],
            [c["accuracy"] for c in mp],
            label=m,
            color=cmap(i % 10),
        )
    ax.set_xlabel("decode throughput (tokens/s, device-synced)")
    ax.set_ylabel("task accuracy")
    ax.set_title("Accuracy vs speed at matched bytes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig_both(fig, args.out)
    print(f"[fig5] wrote {args.out}")


if __name__ == "__main__":
    main()
