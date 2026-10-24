"""Deterministic lexical baseline; no generative models or external services."""

import re
from collections.abc import Sequence

from fixtures import Document


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE))


def search(query: str, documents: Sequence[Document], k: int = 3) -> list[str]:
    if k < 1:
        raise ValueError("k must be positive")
    query_tokens = tokens(query)
    ranked = []
    seen_ids: set[str] = set()
    for doc in documents:
        if doc["id"] in seen_ids:
            raise ValueError("duplicate document ID")
        seen_ids.add(doc["id"])
        score = len(query_tokens & tokens(doc["text"]))
        if score > 0:
            ranked.append((-score, doc["id"]))
    ranked.sort()  # descending score; document ID breaks ties consistently
    return [doc_id for _, doc_id in ranked[:k]]
