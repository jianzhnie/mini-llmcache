# SPDX-License-Identifier: Apache-2.0
"""A 100-sample benchmark dataset, SQuAD-style (long context + questions).

Tries to download SQuAD from Hugging Face first; when the network is
unavailable (as in an air-gapped vLLM container), it falls back to a
locally constructed dataset with the same shape: 20 distinct ~1024-token
contexts, 5 questions each = 100 prompts.  Every question about the same
context shares that context's chunks, which is exactly the reuse pattern
a prefix cache exploits.
"""

import logging

from transformers import PreTrainedTokenizer

from benchmarks.datasets import (
    CHUNK_TOKENS,
    DOCUMENTS,
    SUFFIXES,
    repeat_to_tokens,
)

logger = logging.getLogger(__name__)

N_CONTEXTS = 20
N_QUESTIONS = 5


def _download_squad(tok: PreTrainedTokenizer, context_tokens: int) -> list[dict] | None:
    """Try to pull squad validation from the Hugging Face hub."""
    try:
        import datasets
    except ImportError:
        return None
    try:
        ds = datasets.load_dataset("squad", split="validation")
    except Exception as exc:
        logger.warning("squad download failed (%s); constructing locally", exc)
        return None
    by_context: dict[str, list[str]] = {}
    for row in ds:
        context = " ".join(row["context"].split())
        if len(context) <= 200:
            continue
        by_context.setdefault(context, []).append(row["question"])
        if len(by_context) >= N_CONTEXTS and all(
            len(questions) >= N_QUESTIONS for questions in by_context.values()
        ):
            break
    out = []
    for raw_context, questions in list(by_context.items())[:N_CONTEXTS]:
        context = repeat_to_tokens(tok, raw_context + " ", context_tokens)
        for question in questions[:N_QUESTIONS]:
            out.append({"context": context, "question": question})
    return out if len(out) >= N_CONTEXTS else None


def _construct_local(tok: PreTrainedTokenizer, context_tokens: int) -> list[dict]:
    """Build 100 deterministic samples: 20 contexts x 5 questions.

    Contexts are distinct on purpose: 5 documents x 4 chapter headers, so
    no two contexts share a chunk (different first chunk -> different
    hash chain).
    """
    docs = list(DOCUMENTS.values())
    samples = []
    for i in range(N_CONTEXTS):
        text = docs[i % len(docs)]
        chapter = i // len(docs) + 1
        body = repeat_to_tokens(tok, text, context_tokens)
        context = f"Chapter {chapter}. {body}"
        for q in SUFFIXES[:N_QUESTIONS]:
            samples.append({"context": context, "question": q})
    return samples


def load_100(
    tok: PreTrainedTokenizer,
    context_tokens: int = 4 * CHUNK_TOKENS,
) -> tuple[list[str], list[int]]:
    """Return (prompts, group_ids) for 100 samples.

    Prompts are grouped by shared context: ``group_ids[i]`` is the context
    index of prompt ``i`` (0..19); the first prompt of each group is a
    cold run, the rest hit the context prefix.
    """
    rows = _download_squad(tok, context_tokens) or _construct_local(tok, context_tokens)
    rows = rows[: N_CONTEXTS * N_QUESTIONS]
    prompts, group_ids = [], []
    for gid, row in enumerate(rows):
        prompts.append(f"Context: {row['context']}\n\n{row['question']}")
        group_ids.append(gid // N_QUESTIONS)
    return prompts, group_ids
