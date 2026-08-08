#!/usr/bin/env python3
"""Fig 2 — the correction. Two panels: accuracy vs nominal ratio (left) and accuracy vs realized
bytes (right). Rank inversions between the two views are the paper's central result.

Renders only from real results/*.json.
"""
from __future__ import annotations

import sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _figlib import load_cells, standard_args  # noqa: E402


def main() -> None:
    args = standard_args(__doc__)
    cells = load_cells(args.results)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_nominal: dict[str, list[tuple[float, float]]] = defaultdict(list)
    by_bytes: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for c in cells:
        if "accuracy" not in c:
            continue
        by_nominal[c["method"]].append((c["nominal_ratio"], c["accuracy"]))
        if c.get("realized_bytes"):
            by_bytes[c["method"]].append((c["realized_bytes"], c["accuracy"]))

    if not by_nominal:
        sys.exit("[fig2] no cells carried accuracy — run the sweep first.")

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for method, pts in sorted(by_nominal.items()):
        pts.sort()
        axl.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=method)
    axl.set_xlabel("nominal compression ratio")
    axl.set_ylabel("task accuracy")
    axl.set_title("The field's view (matched tokens)")
    axl.legend(fontsize=8)

    for method, pts in sorted(by_bytes.items()):
        pts.sort()
        axr.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=method)
    axr.set_xlabel("realized KV bytes")
    axr.set_title("Corrected (matched bytes)")

    fig.tight_layout()
    fig.savefig(args.out)
    print(f"[fig2] wrote {args.out}")


if __name__ == "__main__":
    main()
