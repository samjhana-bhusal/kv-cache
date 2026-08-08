# Running kv-bench on Kaggle

`kvbench_kaggle.ipynb` is **self-contained** — it rebuilds the whole `kvbench` package inside the
notebook, so you don't need GitHub or a dataset upload. Regenerate it after any source change with
`python kaggle/build_notebook.py` (it reads the real `src/` and `scripts/`, so it never drifts).

## Steps

1. Go to <https://www.kaggle.com/code> → **New Notebook** → **File → Import Notebook** → upload
   `kaggle/kvbench_kaggle.ipynb`.
2. Right panel → **Settings**:
   - **Accelerator = GPU T4 x2** (or P100).
   - **Internet = On** (needed to pip-install and download model weights).
3. **Run All.** The notebook: installs deps → writes the sources → runs the byte audit (Fig 1) →
   runs the accuracy re-ranking (Fig 2, Table 1) → previews Fig 1 → zips the artifact.
4. **Save Version** (top-right) to persist everything in `/kaggle/working` as notebook output.

Runtime: ~1–3 h for the full 3B+7B sweep. Cell 4 shows how to trim it (3B only, fewer contexts,
smaller `n_items`) for a fast first pass.

## Getting the results back

Everything lands in `/kaggle/working`, which is saved as the notebook's **Output** when you Save
Version. Two ways to reuse it:

- **Download:** Output tab → `kvbench_artifacts.zip` (contains `results/*.json` + `figures/`).
  Unzip into the repo's `results/` and run `cd reports && make paper`, or drop the figures straight
  into Overleaf.
- **Chain notebooks:** Output tab → **New Dataset** from the output, then add that dataset to
  another notebook.

## The result files are the deliverable

`results/*.json` is the versioned artifact (one JSON per cell). The paper's figures regenerate from
them, so keeping the zip is keeping the paper's evidence. Commit the downloaded `results/` back into
the repo so `git` tracks the real numbers.

## Adding Llama-3.1-8B (optional, gated)

1. Accept the license at <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct>.
2. Kaggle → **Add-ons → Secrets** → add `HF_TOKEN`.
3. In the notebook, before the run cells:
   ```python
   from kaggle_secrets import UserSecretsClient
   import os; os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
   !huggingface-cli login --token $HF_TOKEN
   ```
4. Add `- meta-llama/Llama-3.1-8B-Instruct` under `models:` in the config cell.
   `device_map: auto` shards it across the two T4s.
