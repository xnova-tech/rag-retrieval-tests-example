"""Binary relevance metrics for top-k ranked document identifiers."""


def evaluate(ranked: list[str], relevant: set[str], k: int = 3) -> tuple[float, float, float]:
    if not relevant:
        raise ValueError("no relevant documents: evaluate negative queries separately")
    if k < 1:
        raise ValueError("k must be positive")
    selected = ranked[:k]
    if len(selected) != len(set(selected)):
        raise ValueError("duplicate document IDs in ranking")
    hits = sum(doc_id in relevant for doc_id in selected)
    first_rank = next((i for i, doc_id in enumerate(selected, 1) if doc_id in relevant), None)
    return hits / k, hits / len(relevant), 1 / first_rank if first_rank else 0.0
