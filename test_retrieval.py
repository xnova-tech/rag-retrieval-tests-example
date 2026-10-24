"""Arithmetic, ranking, fixture, and regression controls for the example."""

from pathlib import Path

import pytest
from fixtures import Document, Query, load_fixtures
from metrics import evaluate
from retriever import search, tokens

HERE = Path(__file__).resolve().parent

@pytest.fixture(scope="module")
def documents() -> list[Document]:
    return load_fixtures(HERE).documents


@pytest.fixture(scope="module")
def queries() -> list[Query]:
    return load_fixtures(HERE).queries


def test_hand_calculated_metrics() -> None:
    # Two hits at ranks 2 and 3: P=2/3, R=2/2, RR=1/2.
    assert evaluate(["noise", "d1", "d2"], {"d1", "d2"}) == (2 / 3, 1.0, 0.5)


def test_missing_relevant_at_k() -> None:
    assert evaluate(["noise"], {"d1"}) == (0.0, 0.0, 0.0)


def test_no_relevant_requires_separate_check() -> None:
    with pytest.raises(ValueError, match="no relevant documents"):
        evaluate([], set())


def test_duplicate_ranked_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        evaluate(["d1", "d1"], {"d1"})


def test_duplicate_document_id_is_rejected() -> None:
    docs = [{"id": "d1", "text": "alpha"}, {"id": "d1", "text": "alpha"}]
    with pytest.raises(ValueError, match="duplicate document ID"):
        search("alpha", docs)


def test_tie_break_and_order_independence() -> None:
    docs = [{"id": "b", "text": "alpha"}, {"id": "a", "text": "alpha"}]
    assert search("alpha", docs, 3) == ["a", "b"]
    assert search("alpha", list(reversed(docs)), 3) == ["a", "b"]


@pytest.mark.parametrize("k, expected", [(1, ["a"]), (3, ["a", "b"])])
def test_top_k(k: int, expected: list[str]) -> None:
    docs = [{"id": "a", "text": "alpha"}, {"id": "b", "text": "alpha"}]
    assert search("alpha", docs, k) == expected


def test_unicode_casefold_and_missing_query(documents: list[Document]) -> None:
    assert "сброс" in tokens("СБРОС пароля")
    assert search("квантовый двигатель", documents) == []


def test_q03_finds_only_one_of_two_relevant(documents: list[Document]) -> None:
    assert search("выгрузить данные перед удалением", documents) == ["d08"]
    assert evaluate(["d08"], {"d07", "d08"}) == (1 / 3, 1 / 2, 1.0)


def test_q05_relevant_synonym_is_missed(documents: list[Document]) -> None:
    assert search("как вернуть деньги", documents) == []
    assert evaluate([], {"d10"}) == (0.0, 0.0, 0.0)


def test_q06_has_no_gold_and_no_results(
    documents: list[Document], queries: list[Query]
) -> None:
    q06 = next(case for case in queries if case["id"] == "q06")
    assert q06["relevant"] == []
    assert search(str(q06["query"]), documents) == []


def test_regression_detected_when_relevant_falls_below_three() -> None:
    before = evaluate(["d1", "x", "y"], {"d1"})[1]
    after = evaluate(["x", "y", "z", "d1"], {"d1"})[1]
    assert (before, after) == (1.0, 0.0)


def test_fixture_rankings_and_macro(
    documents: list[Document], queries: list[Query]
) -> None:
    expected = {
        "q01": ["d01"],
        "q02": ["d04", "d05", "d06"],
        "q03": ["d08"],
        "q04": ["d06"],
        "q05": [],
        "q06": [],
    }
    scores: list[tuple[float, float, float]] = []
    for case in queries:
        ranking = search(str(case["query"]), documents, k=3)
        assert ranking == expected[case["id"]], case["id"]
        relevant = set(case["relevant"])
        if relevant:
            scores.append(evaluate(ranking, relevant))
    assert len(scores) == 5
    assert [sum(row[i] for row in scores) / 5 for i in range(3)] == pytest.approx(
        [4 / 15, 0.7, 0.8]
    )


def test_fixture_id_integrity(
    documents: list[Document], queries: list[Query]
) -> None:
    ids = [doc["id"] for doc in documents]
    assert len(ids) == len(set(ids)) == 10
    assert len(queries) == 6
    for case in queries:
        assert set(case["relevant"]) <= set(ids)


@pytest.mark.parametrize("operation", ["search", "evaluate"])
def test_invalid_k(operation: str, documents: list[Document]) -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        if operation == "search":
            search("alpha", documents, k=0)
        else:
            evaluate(["a"], {"a"}, k=0)
