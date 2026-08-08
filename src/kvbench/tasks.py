"""Self-contained synthetic long-context task — no dataset download.

A RULER-style needle-in-a-haystack (key/value retrieval): a haystack of filler sentences with one
planted "magic number" fact, and a question at the END asking to recall it. Question-at-end matches
the benchmark convention this project critiques and the kvpress default, so it is the fair setting
for query-aware methods (SnapKV) — we are not stacking the deck against them.

Deterministic given a seed, so runs are reproducible (SRS NFR1). The real extended sweep swaps this
for RULER / LongBench via kvpress's own eval loop; this keeps the local pilot dependency-free.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

FILLER = (
    "The garden was quiet in the afternoon light. "
    "A gentle breeze moved through the tall grass. "
    "Somewhere nearby a bird called out twice. "
    "The old clock on the wall ticked steadily. "
    "Rain had fallen earlier and the ground was damp. "
)


@dataclass
class Needle:
    context: str
    question: str
    answer: str  # the exact string the model must reproduce


def make_needle(
    tokenizer,
    context_tokens: int,
    seed: int = 0,
    depth: float = 0.5,
) -> Needle:
    """Build a needle-in-a-haystack item ~context_tokens long, needle at fractional `depth`."""
    rng = random.Random(seed)
    magic = rng.randint(1000000, 9999999)
    label = rng.choice(["alpha", "bravo", "delta", "orion", "vega", "lyra"])
    needle_sentence = f" The secret code for {label} is {magic}. "

    # Build filler to roughly context_tokens, insert needle at `depth`.
    approx_sentences = max(4, context_tokens // 12)
    body = (FILLER * (approx_sentences // 5 + 1)).split(". ")
    insert_at = int(len(body) * depth)
    body.insert(insert_at, needle_sentence.strip())
    context = ". ".join(s for s in body if s).strip()

    # Trim to token budget.
    ids = tokenizer(context, return_tensors="pt").input_ids[0]
    if ids.shape[0] > context_tokens:
        ids = ids[:context_tokens]
        context = tokenizer.decode(ids, skip_special_tokens=True)
        if str(magic) not in context:
            # needle got trimmed; re-insert near the (now shorter) middle
            context = context[: len(context) // 2] + needle_sentence + context[len(context) // 2 :]

    question = f"What is the secret code for {label}? Answer with the number only."
    return Needle(context=context, question=question, answer=str(magic))


def score(prediction: str, answer: str) -> float:
    """1.0 if the exact answer string appears in the prediction, else 0.0."""
    return 1.0 if answer in prediction else 0.0
