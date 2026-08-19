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


def _download_squad() -> list[dict] | None:
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
    rows = [r for r in ds if len(r["context"]) > 200][:N_CONTEXTS]
    out = []
    for r in rows:
        context = " ".join(r["context"].split())[:8000]
        for q in r["questions"][:N_QUESTIONS]:
            out.append({"context": context, "question": q})
    return out if len(out) >= N_CONTEXTS else None


def _construct_local(tok: PreTrainedTokenizer) -> list[dict]:
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
        body = repeat_to_tokens(tok, text, 4 * CHUNK_TOKENS)
        context = f"Chapter {chapter}. {body}"
        for q in SUFFIXES[:N_QUESTIONS]:
            samples.append({"context": context, "question": q})
    return samples


def load_100(tok: PreTrainedTokenizer) -> tuple[list[str], list[int]]:
    """Return (prompts, group_ids) for 100 samples.

    Prompts are grouped by shared context: ``group_ids[i]`` is the context
    index of prompt ``i`` (0..19); the first prompt of each group is a
    cold run, the rest hit the context prefix.
    """
    rows = _download_squad() or _construct_local(tok)
    rows = rows[: N_CONTEXTS * N_QUESTIONS]
    prompts, group_ids = [], []
    for gid, row in enumerate(rows):
        prompts.append(f"Context: {row['context']}\n\n{row['question']}")
        group_ids.append(gid // N_QUESTIONS)
    return prompts, group_ids
