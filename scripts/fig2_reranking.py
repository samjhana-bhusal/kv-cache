#!/usr/bin/env python3
"""Fig 2 — the correction. Two panels from the same accuracy cells:
   LEFT : accuracy vs nominal compression ratio (the field's view).
   RIGHT: accuracy vs realized KV bytes / full cache (the honest view).

Simulation-only methods (AdaKV) look like normal compressors on the left but collapse onto x~1.0
on the right, exposing that they never freed memory. Renders only from real results/acc__*.json.
"""
from __future__ import annotations

import sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _figlib import load_cells, standard_args, filter_model, savefig_both  # noqa: E402


def main() -> None:
    args = standard_args(__doc__)
    cells = [c for c in filter_model(load_cells(args.results), args.model)
             if c.get("kind") == "accuracy" and not c.get("failed")]
    if not cells:
        sys.exit("[fig2] no accuracy cells (results/acc__*.json) — run kvbench.eval first.")
    # Prefer RULER (standard benchmark) over synthetic NIAH when both are present.
    if any(c.get("task") == "ruler" for c in cells):
        cells = [c for c in cells if c.get("task") == "ruler"]
        print("[fig2] using RULER cells")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ctx = max(c["context_len"] for c in cells)
    cells = [c for c in cells if c["context_len"] == ctx]

    by_nominal: dict[str, list[tuple[float, float]]] = defaultdict(list)
    by_bytes: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for c in cells:
        by_nominal[c["method"]].append((c["nominal_ratio"], c["accuracy"]))
        if c.get("realized_fraction") is not None:
            by_bytes[c["method"]].append((c["realized_fraction"], c["accuracy"]))

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for method, pts in sorted(by_nominal.items()):
        pts.sort()
        axl.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=method)
    axl.set_xlabel("nominal compression ratio")
    axl.set_ylabel("task accuracy")
    axl.set_title("The field's view (matched nominal ratio)")
    axl.legend(fontsize=8)

    for method, pts in sorted(by_bytes.items()):
        pts.sort()
        axr.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=method)
    axr.axvline(1.0, color="0.6", lw=0.8, ls=":")
    axr.set_xlabel("realized KV bytes / full cache")
    axr.set_title("Corrected (matched realized bytes)")

    fig.suptitle(f"Ratios vs bytes (ctx={ctx})")
    fig.tight_layout()
    savefig_both(fig, args.out)
    print(f"[fig2] wrote {args.out} from {len(cells)} accuracy cells")


if __name__ == "__main__":
    main()
