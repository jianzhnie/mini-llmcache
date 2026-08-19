# SPDX-License-Identifier: Apache-2.0
"""Deterministic benchmark prompt datasets for mini-llmcache.

Prompts are built from a fixed phrase repeated to an exact token count
(the cache chunks tokens in groups of 256, so prompt sizes are expressed
in chunks of 256 tokens).  Everything is deterministic: identical runs
produce identical prompts and timings are comparable across versions.

The three scenarios mirror the three cache-reuse patterns worth measuring:

1. ``exact_repeat``  — the same prompt sent again (full hit);
2. ``shared_prefix`` — many prompts share one long prefix (partial hit,
   e.g. a shared system prompt / document in RAG or multi-turn chat);
3. ``no_reuse``      — distinct prompts of equal length (cold baseline).
"""

from transformers import PreTrainedTokenizer

PHRASE = "A field guide to the birds of North America. "
CHUNK_TOKENS = 256

#: Documents used as prefixes in the shared-prefix scenario.  Each is
#: repeated to the target token count; distinct documents never share
#: chunks with each other.  Scenarios must use *different* documents so
#: they do not contaminate one another's cache state.
DOCUMENTS = {
    "birds": "A field guide to the birds of North America. ",
    "fungi": "Mushrooms of the Pacific Northwest: a forager's handbook. ",
    "rocks": "The geology of the Appalachian Mountains, volume two. ",
    "ships": "An illustrated history of clipper ships and the tea trade. ",
    "weather": "Cloud atlas: a photographic study of storm systems. ",
    "insects": "A field guide to the dragonflies and damselflies of Texas. ",
    "plants": "Edible wild plants of the Rocky Mountain region. ",
}

#: Short questions appended after a shared prefix (the varying tail).
SUFFIXES = [
    "Question: What is the main subject of this document? Answer:",
    "Question: Summarize the first paragraph in one sentence. Answer:",
    "Question: How many species are mentioned? Answer:",
    "Question: Translate the title into French. Answer:",
    "Question: List the key terms in order of appearance. Answer:",
]


def repeat_to_tokens(tok: PreTrainedTokenizer, text: str, target_tokens: int) -> str:
    """Repeat ``text`` until the encoding reaches ``target_tokens``,
    then trim to exactly that many tokens (deterministic)."""
    per = len(tok.encode(text))
    times = max(1, target_tokens // per + 1)
    encoded = tok.encode(text * times)[:target_tokens]
    return tok.decode(encoded)


def make_exact_repeat(
    tok: PreTrainedTokenizer, n_chunks: int = 4, doc: str = "ships"
) -> dict:
    """One prompt of ``n_chunks`` chunks; send it twice in the benchmark."""
    prompt = repeat_to_tokens(tok, DOCUMENTS[doc], n_chunks * CHUNK_TOKENS)
    return {"prompts": [prompt], "tokens": n_chunks * CHUNK_TOKENS}


def make_shared_prefix(
    tok: PreTrainedTokenizer,
    prefix_chunks: int = 8,
    n_suffixes: int = 4,
    doc: str = "birds",
) -> dict:
    """A common prefix plus ``n_suffixes`` different tails.

    Returns the prefix, the prompts (prefix + each suffix), and a
    ``baseline`` prompt of the same total length built from a *different*
    document (never cached, used as the cold reference).
    """
    prefix = repeat_to_tokens(tok, DOCUMENTS[doc], prefix_chunks * CHUNK_TOKENS)
    prompts = [prefix + " " + s for s in SUFFIXES[:n_suffixes]]
    baseline = repeat_to_tokens(
        tok,
        DOCUMENTS["rocks"],
        prefix_chunks * CHUNK_TOKENS + len(tok.encode(SUFFIXES[0])),
    )
    return {
        "prefix": prefix,
        "prompts": prompts,
        "baseline": baseline,
        "prefix_tokens": prefix_chunks * CHUNK_TOKENS,
    }


def make_no_reuse(
    tok: PreTrainedTokenizer,
    n_chunks: int = 4,
    n_prompts: int = 4,
    docs: tuple[str, ...] = ("fungi", "rocks", "ships", "weather"),
) -> dict:
    """``n_prompts`` distinct same-length prompts (all cold)."""
    prompts = [
        repeat_to_tokens(tok, DOCUMENTS[docs[i % len(docs)]], n_chunks * CHUNK_TOKENS)
        for i in range(n_prompts)
    ]
    return {"prompts": prompts, "tokens": n_chunks * CHUNK_TOKENS}
