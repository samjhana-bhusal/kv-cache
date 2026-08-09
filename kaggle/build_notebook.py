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
    "src/kvbench/longbench.py",
    "src/kvbench/selftest.py",
    "scripts/_figlib.py",
    "scripts/fig1_mispricing.py",
    "scripts/fig2_reranking.py",
    "scripts/fig3_decomposition.py",
    "scripts/fig5_latency.py",
    "scripts/table1_rank.py",
    "scripts/cross_model_summary.py",
    "configs/kaggle.yaml",
    "configs/kaggle_ruler.yaml",
    "configs/kaggle_full.yaml",
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


def _flush_cell() -> dict:
    """A GPU-flush cell: free the model, collect, empty the CUDA cache, and show it's reclaimed."""
    return code_cell(
        "import gc, torch\n"
        "for _n in ['model','pipe','m']:\n"
        "    if _n in globals(): del globals()[_n]\n"
        "gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()\n"
        "free,total=torch.cuda.mem_get_info()\n"
        "print(f'GPU flushed — {free/1e9:.1f} GB free of {total/1e9:.1f} GB')\n"
    )


def build_full() -> dict:
    """Second-model, all-experiments notebook: audit(+decomposition) + RULER + LongBench(+latency)
    on Mistral-7B, with explicit GPU flushes between passes. Addresses the two reviewer asks."""
    cells = []
    cells.append(md_cell(
        "# kv-bench — second model + all experiments (*Bytes, Not Ratios*)\n"
        "\n"
        "Addresses the two reviewer asks in one run: a **second model family** (Mistral-7B,\n"
        "GQA group 4 — different from Qwen's group 8) and the two previously-unfinished measurements\n"
        "(**byte decomposition** and **decode latency**), plus a **second benchmark** (LongBench QA).\n"
        "\n"
        "**Settings → Accelerator = GPU T4 x2, Internet = On → Run All.** Each experiment below runs\n"
        "as its own process; the GPU is explicitly flushed between them (you'll see free-memory\n"
        "printouts). Mistral-7B is open (no gating). Budget ~2–4 h for the full run; trim in the\n"
        "config cell for a fast check. Results land in `/kaggle/working` and zip to an artifact."
    ))

    cells.append(md_cell("## 1. Install"))
    cells.append(code_cell(
        "!pip install -q kvpress matplotlib pyyaml\n"
        "import torch\n"
        "print('CUDA:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())\n"
        "assert torch.cuda.is_available(), 'Enable GPU: Settings -> Accelerator -> GPU'\n"
    ))

    cells.append(md_cell("## 2. Write the kvbench sources"))
    cells.append(code_cell(
        "import os\n"
        "for d in ['src/kvbench','scripts','configs','results','reports/figures']:\n"
        "    os.makedirs(f'/kaggle/working/{d}', exist_ok=True)\n"
    ))
    for rel in MODULES:
        cells.append(writefile_cell(rel))

    cells.append(md_cell("## 3. Sanity: byte instrument"))
    cells.append(code_cell(
        "import os; os.chdir('/kaggle/working')\n"
        "!PYTHONPATH=/kaggle/working/src python -m kvbench.selftest\n"
    ))

    cells.append(md_cell(
        "## 4a. (Optional) Hugging Face token — only needed for gated models (Llama-3.1-8B)\n"
        "Mistral-7B is open and needs no token. For Llama: add your token under **Add-ons → Secrets**\n"
        "as `HF_TOKEN`, then run this cell. Skip it otherwise."
    ))
    cells.append(code_cell(
        "try:\n"
        "    from kaggle_secrets import UserSecretsClient\n"
        "    import os\n"
        "    os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')\n"
        "    from huggingface_hub import login; login(os.environ['HF_TOKEN'])\n"
        "    print('HF token set — gated models enabled')\n"
        "except Exception as e:\n"
        "    print('No HF token (fine for open models like Mistral):', e)\n"
    ))
    cells.append(md_cell(
        "## 4b. (Optional) trim the config\n"
        "Default: Mistral-7B, ctx 4096, 50 items/cell. Edit here to add models (each runs\n"
        "sequentially — one model on the GPU at a time, sharded across both T4s), shrink `n_items`\n"
        "for a fast first run, or set `max_context_length` (LongBench contexts reach 30k+ tokens and\n"
        "are capped to this to fit the GPU; default = the context length)."
    ))
    cells.append(code_cell(
        "import yaml\n"
        "cfg = yaml.safe_load(open('/kaggle/working/configs/kaggle_full.yaml'))\n"
        "# cfg['models'] = ['mistralai/Mistral-7B-Instruct-v0.3', 'meta-llama/Llama-3.1-8B-Instruct']\n"
        "# cfg['n_items'] = 20\n"
        "cfg.setdefault('max_context_length', 4096)   # LongBench safety cap\n"
        "yaml.safe_dump(cfg, open('/kaggle/working/configs/kaggle_full.yaml','w'))\n"
        "print(open('/kaggle/working/configs/kaggle_full.yaml').read())\n"
    ))

    cells.append(md_cell(
        "## 5. Byte audit + decomposition (Fig 1, Fig 3)\n"
        "Records realized bytes, the memory-faithful/simulation-only/incommensurable taxonomy, and\n"
        "now the per-method byte decomposition (payload / metadata / peak-prefill transient) on the\n"
        "second model family."
    ))
    cells.append(code_cell(
        "!CUDA_LAUNCH_BLOCKING=1 PYTHONPATH=/kaggle/working/src python -m kvbench.run "
        "--config configs/kaggle_full.yaml --results results --device cuda\n"
    ))
    cells.append(md_cell("### flush GPU before the next experiment"))
    cells.append(_flush_cell())

    cells.append(md_cell(
        "## 6. RULER accuracy + latency (Fig 2, Fig 5)\n"
        "Byte-matched re-ranking on RULER, now also recording device-synced generation throughput."
    ))
    cells.append(code_cell(
        "!CUDA_LAUNCH_BLOCKING=1 PYTHONPATH=/kaggle/working/src python -m kvbench.eval "
        "--config configs/kaggle_full.yaml --task ruler --results results --device cuda\n"
    ))
    cells.append(md_cell("### flush GPU"))
    cells.append(_flush_cell())

    cells.append(md_cell(
        "## 7. LongBench QA accuracy + latency (second benchmark)\n"
        "Natural-text QA subsets scored with token-F1 — the second benchmark reviewers asked for.\n"
        "Downloads LongBench on first use."
    ))
    cells.append(code_cell(
        "!CUDA_LAUNCH_BLOCKING=1 PYTHONPATH=/kaggle/working/src python -m kvbench.eval "
        "--config configs/kaggle_full.yaml --task longbench --results results --device cuda\n"
    ))
    cells.append(md_cell("### flush GPU"))
    cells.append(_flush_cell())

    cells.append(md_cell(
        "## 8. Generate figures (per model) + cross-model summary\n"
        "Each figure is written both as `.pdf` (for the paper) and `.png` (to view here). With\n"
        "multiple models, per-model figures are emitted (`fig1_<model>.pdf`), plus a cross-model\n"
        "summary table — the figure-independent evidence that the taxonomy holds across families."
    ))
    cells.append(code_cell(
        "import glob, json, os, subprocess\n"
        "ENV = dict(os.environ, PYTHONPATH='/kaggle/working/src')\n"
        "def sh(*a): subprocess.run(a, env=ENV, cwd='/kaggle/working')\n"
        "cells = [json.load(open(p)) for p in glob.glob('results/*.json')]\n"
        "models = sorted({c.get('model') for c in cells if c.get('model')})\n"
        "print('models present:', models)\n"
        "for m in models:\n"
        "    tag = m.split('/')[-1]\n"
        "    for s in ['fig1_mispricing','fig2_reranking','fig3_decomposition','fig5_latency']:\n"
        "        sh('python', f'scripts/{s}.py', '--results', 'results', '--model', tag,\n"
        "           '--out', f'reports/figures/{s}__{tag}.pdf')\n"
        "    sh('python', 'scripts/table1_rank.py', '--results', 'results', '--model', tag,\n"
        "       '--out', f'reports/figures/table1_rank__{tag}.tex')\n"
        "sh('python', 'scripts/cross_model_summary.py', '--results', 'results',\n"
        "   '--out', 'reports/figures/cross_model_summary.tex')\n"
        "print(open('reports/figures/cross_model_summary.tex').read())\n"
    ))
    cells.append(md_cell("### Preview the cross-model summary + one model's figures (PNG)"))
    cells.append(code_cell(
        "from IPython.display import Image, display\n"
        "import glob\n"
        "for png in sorted(glob.glob('reports/figures/*.png'))[:8]:\n"
        "    print(png); display(Image(png))\n"
    ))

    cells.append(md_cell("## 9. Package artifact"))
    cells.append(code_cell(
        "import shutil, glob\n"
        "os.makedirs('/kaggle/working/artifact', exist_ok=True)\n"
        "shutil.copytree('/kaggle/working/results','/kaggle/working/artifact/results',dirs_exist_ok=True)\n"
        "shutil.copytree('/kaggle/working/reports/figures','/kaggle/working/artifact/figures',dirs_exist_ok=True)\n"
        "shutil.make_archive('/kaggle/working/kvbench_artifacts_full','zip','/kaggle/working/artifact')\n"
        "print(len(glob.glob('/kaggle/working/results/*.json')),'cells -> kvbench_artifacts_full.zip')\n"
    ))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}, "accelerator": "GPU",
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def main() -> None:
    for builder, name in [(build, "kvbench_kaggle.ipynb"), (build_full, "kvbench_full.ipynb")]:
        nb = builder()
        out = os.path.join(ROOT, "kaggle", name)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print(f"wrote {out} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
