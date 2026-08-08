"""RULER task adapter — the credible long-context benchmark, replacing synthetic NIAH for the
headline accuracy claim.

Loads NVIDIA RULER via the HuggingFace dataset `simonjegou/ruler` (the same one kvpress's own eval
uses), keyed by context length, and scores with RULER's official metric (substring match; `qa_*`
tasks use match-any, the rest match-all). Using the standard benchmark + standard metric removes the
recency-bias caveat of our synthetic needle task.

RULER places the question at the END of the context (question-last), the convention this project
critiques — so it is the fair, unmodified setting for query-aware methods (SnapKV).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from datasets import load_dataset

_CTRL = re.compile(r"[\x00-\x1f]")


@dataclass
class RulerItem:
    context: str
    question: str
    answer_prefix: str
    answers: list[str]   # RULER stores a list of acceptable reference strings
    task: str
    max_new_tokens: int


def load_ruler(context_len: int, n_items: int, seed: int = 0) -> list[RulerItem]:
    """Load up to n_items RULER rows for a given context length, shuffled deterministically.

    context_len must be a RULER config (4096, 8192, 16384, ...). We sample across all 13 tasks so
    the accuracy number is not dominated by one task type.
    """
    ds = load_dataset("simonjegou/ruler", data_dir=str(context_len), split="test")
    ds = ds.shuffle(seed=seed)
    n = min(n_items, len(ds))
    items = []
    for r in ds.select(range(n)):
        ans = r["answer"]
        if isinstance(ans, str):
            ans = [ans]
        items.append(
            RulerItem(
                context=r["context"],
                question=r["question"],
                answer_prefix=r["answer_prefix"],
                answers=list(ans),
                task=r["task"],
                max_new_tokens=int(r["max_new_tokens"]),
            )
        )
    return items


def score_item(prediction: str, item: RulerItem) -> float:
    """RULER official metric per item: substring match of reference(s) in the prediction.

    `qa_*` tasks score match-any (any reference present -> 1.0); all other tasks score match-all
    (fraction of references present). Mirrors kvpress benchmarks/ruler/calculate_metrics.py.
    """
    pred = _CTRL.sub("", prediction.strip()).strip().lower()
    refs = [a.lower() for a in item.answers]
    if not refs:
        return 0.0
    hits = [1.0 if r in pred else 0.0 for r in refs]
    if item.task.split("_")[0] == "qa":
        return max(hits)          # match-any
    return sum(hits) / len(hits)  # match-all
