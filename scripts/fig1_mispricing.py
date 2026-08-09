#!/usr/bin/env python3
"""Fig 1 — nominal compression ratio (x) vs realized KV bytes as a fraction of full cache (y).

The paper's motivating figure. Memory-faithful methods track the diagonal; simulation-only methods
(e.g. masked AdaKV) render as flat lines at ~1.0; sink/observation-window methods diverge upward.
Renders only from real results/*.json — see _figlib.load_cells.
"""
from __future__ import annotations

import sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _figlib import load_cells, standard_args, filter_model, savefig_both  # noqa: E402


def main() -> None:
    args = standard_args(__doc__)
    cells = filter_model(load_cells(args.results), args.model)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = [c for c in cells if not c.get("failed")]
    # One model, one context length for a clean figure: use the largest context present.
    ctx = max(c["context_len"] for c in cells if "context_len" in c)
    cells = [c for c in cells if c.get("context_len") == ctx]

    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for c in cells:
        full = c.get("full_cache_bytes")
        if not full:
            continue
        series[c["method"]].append((c["nominal_ratio"], c["realized_bytes"] / full))

    if not series:
        sys.exit("[fig1] no cells carried realized_bytes/full_cache_bytes — cannot plot.")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ratio promises (payload = fraction)")
    for method, pts in sorted(series.items()):
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", label=method)
    ax.set_xlabel("nominal compression ratio (fraction of tokens retained)")
    ax.set_ylabel("realized KV bytes / full-cache bytes")
    ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
    ymax = max(1.12, max(y for pts in series.values() for _, y in pts) + 0.03)
    ax.set_ylim(0, ymax)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title(f"Ratios are not bytes (ctx={ctx})")
    fig.tight_layout()
    savefig_both(fig, args.out)
    print(f"[fig1] wrote {args.out} from {len(cells)} cells, {len(series)} methods")


if __name__ == "__main__":
    main()
