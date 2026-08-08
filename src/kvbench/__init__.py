"""kvbench — byte-accounting audit of KV cache compression methods.

The thesis: the field budgets compression in tokens and reports it as memory savings; those are
not the same quantity. This package measures realized bytes directly (bytes.py), classifies each
method into a memory-faithfulness taxonomy (presses.py), and drives a resumable sweep (run.py).

Built on NVIDIA kvpress (vendored) + HuggingFace transformers >=5. Device-agnostic: MPS today,
CUDA on the Ubuntu rerun.
"""

__version__ = "0.1.0"
