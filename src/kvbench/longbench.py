"""LongBench task adapter — a second, realistic long-context benchmark alongside RULER.

Loads the English QA subsets of LongBench (Xnhyacinth/LongBench, the dataset kvpress's eval uses),
which all score with token-level F1 (SQuAD-style). Using one metric across the QA subsets keeps the
number comparable and self-contained: F1 is reimplemented here (standard normalization) so the
notebook needs no extra scoring dependencies (jieba/rouge/fuzzywuzzy).

LongBench places the question last, like RULER — the fair, unmodified setting for query-aware
methods. It is a natural-text complement to RULER's synthetic tasks, directly addressing the
"more benchmarks" reviewer ask.
"""
from __future__ import annotations

import ast
import re
import string
from collections import Counter
from dataclasses import dataclass

from datasets import concatenate_datasets, load_dataset

# English QA subsets: all use token-F1. (Summarization/code/counting subsets use other metrics and
# are omitted to keep a single comparable score.)
QA_TASKS = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique"]


@dataclass
class LongBenchItem:
    context: str
    question: str
    answer_prefix: str
    answers: list[str]
    task: str
    max_new_tokens: int


def _parse_answers(a) -> list[str]:
    if isinstance(a, list):
        return [str(x) for x in a]
    if isinstance(a, str):
        try:
            v = ast.literal_eval(a)
            return [str(x) for x in v] if isinstance(v, list) else [a]
        except (ValueError, SyntaxError):
            return [a]
    return [str(a)]


def load_longbench(n_items: int, tasks: list[str] | None = None, seed: int = 0) -> list[LongBenchItem]:
    """Sample up to n_items across the English QA subsets, shuffled deterministically."""
    tasks = tasks or QA_TASKS
    parts = []
    for t in tasks:
        try:
            parts.append(load_dataset("Xnhyacinth/LongBench", t, split="test"))
        except Exception:  # noqa: BLE001 — skip a subset that fails to load rather than abort
            continue
    if not parts:
        raise RuntimeError("LongBench: no QA subsets loaded")
    ds = concatenate_datasets(parts).shuffle(seed=seed)
    n = min(n_items, len(ds))
    items = []
    for r in ds.select(range(n)):
        items.append(
            LongBenchItem(
                context=r["context"],
                question=r["question"],
                answer_prefix=r.get("answer_prefix", "Answer:"),
                answers=_parse_answers(r["answers"]),
                task=r["dataset"] if "dataset" in r else r.get("task", "qa"),
                max_new_tokens=int(r.get("max_new_tokens", 64)),
            )
        )
    return items


def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gt_tokens = _normalize(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def score_item(prediction: str, item: LongBenchItem) -> float:
    """Token-level F1, max over reference answers (SQuAD/LongBench-QA convention)."""
    if not item.answers:
        return 0.0
    return max(_f1(prediction, a) for a in item.answers)
