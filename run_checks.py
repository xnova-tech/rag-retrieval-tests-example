"""Deterministic integration check with exact, reproducible output."""

# ruff: noqa: E402, T201

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # needed with python -I
from fixtures import load_fixtures
from metrics import evaluate
from retriever import search


def main() -> None:
    fixtures = load_fixtures(HERE)
    documents, queries = fixtures.documents, fixtures.queries
    doc_ids = [doc["id"] for doc in documents]
    if len(doc_ids) != len(set(doc_ids)) or len(doc_ids) != 10:
        raise ValueError("expected ten documents with unique IDs")
    if len(queries) != 6:
        raise ValueError("expected six queries")
    all_scores: list[tuple[float, float, float]] = []
    for case in queries:
        ranked = search(case["query"], documents, k=3)
        relevant = set(case["relevant"])
        if not relevant <= set(doc_ids):
            raise ValueError(f"unknown relevant document in {case['id']}")
        if not relevant:
            if ranked:
                raise ValueError(f"negative query {case['id']} returned {ranked}")
            print(f"{case['id']}: top3=[] no_relevant_documents")
            continue
        precision, recall, reciprocal_rank = evaluate(ranked, relevant, k=3)
        all_scores.append((precision, recall, reciprocal_rank))
        print(
            f"{case['id']}: top3={','.join(ranked) or '-'} "
            f"P@3={precision:.3f} R@3={recall:.3f} RR@3={reciprocal_rank:.3f}"
        )
    means = [sum(row[j] for row in all_scores) / len(all_scores) for j in range(3)]
    print(f"macro(5): P@3={means[0]:.3f} R@3={means[1]:.3f} MRR@3={means[2]:.3f}")


if __name__ == "__main__":
    main()
