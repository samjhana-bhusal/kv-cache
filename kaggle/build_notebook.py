#!/usr/bin/env python3
"""Generate a self-contained Kaggle notebook from the real kvbench sources.

Single source of truth: reads src/kvbench/*.py, scripts/*.py, and configs/kaggle.yaml, and emits
kaggle/kvbench_kaggle.ipynb whose setup cells recreate the exact repo layout under /kaggle/working.
The user uploads the .ipynb to Kaggle, enables GPU + Internet, and Runs All. No GitHub, no dataset
upload. Results and figures land in /kaggle/working (persisted as notebook output) and are also
zipped into an artifact for one-click download.

Regenerate after any source change:  python kaggle/build_notebook.py
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODULES = [
    "src/kvbench/__init__.py",
    "src/kvbench/device.py",
    "src/kvbench/bytes.py",
    "src/kvbench/presses.py",
    "src/kvbench/tasks.py",
    "src/kvbench/run.py",
    "src/kvbench/eval.py",
    "src/kvbench/ruler.py",
    "src/kvbench/selftest.py",
    "scripts/_figlib.py",
    "scripts/fig1_mispricing.py",
    "scripts/fig2_reranking.py",
    "scripts/fig3_decomposition.py",
    "scripts/fig5_latency.py",
    "scripts/table1_rank.py",
    "configs/kaggle.yaml",
    "configs/kaggle_ruler.yaml",
]


def code_cell(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src if isinstance(src, list) else src.splitlines(keepends=True),
    }


def md_cell(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src if isinstance(src, list) else src.splitlines(keepends=True),
    }


def writefile_cell(relpath: str) -> dict:
    with open(os.path.join(ROOT, relpath), encoding="utf-8") as f:
        content = f.read()
    target = f"/kaggle/working/{relpath}"
    header = f"%%writefile {target}\n"
    return code_cell([header] + content.splitlines(keepends=True))


def build() -> dict:
    cells = []

    cells.append(md_cell(
        "# kv-bench on Kaggle — *Bytes, Not Ratios*\n"
        "\n"
        "Self-contained. **Before running:** open the right-hand panel →\n"
        "**Settings → Accelerator = GPU T4 x2** (or P100) and **Internet = On**.\n"
        "Then **Run All**. Results + figures are written to `/kaggle/working` (saved as output when\n"
        "you *Save Version*) and zipped into `kvbench_artifacts.zip` for download.\n"
        "\n"
        "Runtime: roughly 1–3 h for the full 3B+7B sweep. To shorten, edit `configs/kaggle.yaml`\n"
        "(fewer models / context lengths / `n_items`) in the config cell below before running.\n"
    ))

    cells.append(md_cell("## 1. Install"))
    cells.append(code_cell(
        "# kvpress pulls a compatible torch/transformers. Quiet install.\n"
        "!pip install -q kvpress matplotlib pyyaml\n"
        "import torch\n"
        "print('CUDA:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())\n"
        "assert torch.cuda.is_available(), 'Enable GPU: Settings -> Accelerator -> GPU'\n"
    ))

    cells.append(md_cell("## 2. Write the kvbench sources (exact repo layout)"))
    cells.append(code_cell(
        "import os\n"
        "for d in ['src/kvbench', 'scripts', 'configs', 'results', 'reports/figures']:\n"
        "    os.makedirs(f'/kaggle/working/{d}', exist_ok=True)\n"
    ))
    for rel in MODULES:
        cells.append(writefile_cell(rel))

    cells.append(md_cell(
        "## 3. Sanity: the byte instrument must pass\n"
        "All three controls, including the masking positive control, must say PASS."
    ))
    cells.append(code_cell(
        "import os; os.chdir('/kaggle/working')\n"
        "!PYTHONPATH=/kaggle/working/src python -m kvbench.selftest\n"
    ))

    cells.append(md_cell(
        "## 4. (Optional) edit the config\n"
        "Uncomment to trim the sweep for a faster first run."
    ))
    cells.append(code_cell(
        "# import yaml\n"
        "# cfg = yaml.safe_load(open('/kaggle/working/configs/kaggle.yaml'))\n"
        "# cfg['models'] = ['Qwen/Qwen2.5-3B-Instruct']   # 3B only\n"
        "# cfg['context_lengths'] = [2048]\n"
        "# cfg['n_items'] = 10\n"
        "# yaml.safe_dump(cfg, open('/kaggle/working/configs/kaggle.yaml','w'))\n"
        "# print(open('/kaggle/working/configs/kaggle.yaml').read())\n"
    ))

    cells.append(md_cell(
        "## 5. Phase 2 — byte audit (Fig 1). No accuracy eval; the motivating result."
    ))
    cells.append(code_cell(
        "# CUDA_LAUNCH_BLOCKING=1 makes any CUDA error report at the real kernel (accurate traceback).\n"
        "!CUDA_LAUNCH_BLOCKING=1 PYTHONPATH=/kaggle/working/src python -m kvbench.run "
        "--config configs/kaggle.yaml --results results --device cuda\n"
    ))

    cells.append(md_cell(
        "## 6. Phase 3 — accuracy re-ranking at matched bytes (Fig 2, Table 1)"
    ))
    cells.append(code_cell(
        "!CUDA_LAUNCH_BLOCKING=1 PYTHONPATH=/kaggle/working/src python -m kvbench.eval "
        "--config configs/kaggle.yaml --results results --device cuda\n"
    ))

    cells.append(md_cell(
        "## 6b. (Optional but recommended) RULER — the credible benchmark\n"
        "Re-runs the accuracy re-ranking on real **RULER** (`simonjegou/ruler`) with RULER's official\n"
        "metric, removing the recency-bias caveat of the synthetic needle task. Downloads the RULER\n"
        "dataset (a few GB) on first use; skip this cell if you only want the byte-audit + synthetic\n"
        "accuracy. RULER cells (`rul__*`) automatically supersede the synthetic ones in the figures."
    ))
    cells.append(code_cell(
        "!CUDA_LAUNCH_BLOCKING=1 PYTHONPATH=/kaggle/working/src python -m kvbench.eval "
        "--config configs/kaggle_ruler.yaml --results results --device cuda\n"
    ))

    cells.append(md_cell("## 7. Generate figures + table from the real results"))
    cells.append(code_cell(
        "!PYTHONPATH=/kaggle/working/src python scripts/fig1_mispricing.py "
        "--results results --out reports/figures/fig1_mispricing.pdf\n"
        "!PYTHONPATH=/kaggle/working/src python scripts/fig2_reranking.py "
        "--results results --out reports/figures/fig2_reranking.pdf\n"
        "!PYTHONPATH=/kaggle/working/src python scripts/table1_rank.py "
        "--results results --out reports/figures/table1_rank.tex\n"
        "print(open('reports/figures/table1_rank.tex').read())\n"
    ))

    cells.append(md_cell("## 8. Preview Fig 1 inline"))
    cells.append(code_cell(
        "import matplotlib.pyplot as plt, matplotlib.image as mpimg, json, glob\n"
        "from collections import defaultdict\n"
        "cells=[json.load(open(p)) for p in glob.glob('results/*.json')]\n"
        "cells=[c for c in cells if not c.get('failed') and c.get('kind')!='accuracy']\n"
        "ctx=max(c['context_len'] for c in cells)\n"
        "cells=[c for c in cells if c['context_len']==ctx]\n"
        "s=defaultdict(list)\n"
        "for c in cells: s[c['method']].append((c['nominal_ratio'], c['realized_bytes']/c['full_cache_bytes']))\n"
        "fig,ax=plt.subplots(figsize=(7,4.3))\n"
        "ax.plot([0,1],[1,0],'k--',lw=1,label='ratio promises'); ax.axhline(1.0,color='0.6',ls=':')\n"
        "for m,p in sorted(s.items()):\n"
        "    p.sort(); ax.plot([x for x,_ in p],[y for _,y in p],marker='o',label=m)\n"
        "ax.set_xlabel('nominal compression ratio'); ax.set_ylabel('realized bytes / full cache')\n"
        "ax.set_title(f'Ratios are not bytes (ctx={ctx})'); ax.legend(fontsize=8,loc='center left',bbox_to_anchor=(1.01,0.5))\n"
        "plt.tight_layout(); plt.show()\n"
    ))

    cells.append(md_cell(
        "## 9. Package results as a downloadable artifact\n"
        "Everything in `/kaggle/working` persists when you **Save Version**. This also zips the\n"
        "results + figures so you can download one file (Output tab → `kvbench_artifacts.zip`), or\n"
        "**Create Dataset** from the output to reuse in another notebook."
    ))
    cells.append(code_cell(
        "import shutil, glob, os\n"
        "os.makedirs('/kaggle/working/artifact', exist_ok=True)\n"
        "shutil.copytree('/kaggle/working/results', '/kaggle/working/artifact/results', dirs_exist_ok=True)\n"
        "shutil.copytree('/kaggle/working/reports/figures', '/kaggle/working/artifact/figures', dirs_exist_ok=True)\n"
        "shutil.make_archive('/kaggle/working/kvbench_artifacts', 'zip', '/kaggle/working/artifact')\n"
        "n=len(glob.glob('/kaggle/working/results/*.json'))\n"
        "print(f'{n} result cells zipped -> /kaggle/working/kvbench_artifacts.zip')\n"
    ))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    nb = build()
    out = os.path.join(ROOT, "kaggle", "kvbench_kaggle.ipynb")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"wrote {out} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
