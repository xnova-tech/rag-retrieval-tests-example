"""Load and validate the small, fixed retrieval evaluation corpus."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

JsonRow = dict[str, object]
DIRECTORY = Path(__file__).resolve().parent


class Document(TypedDict):
    id: str
    text: str


class Query(TypedDict):
    id: str
    query: str
    relevant: list[str]
    reason: str


@dataclass(frozen=True)
class Fixtures:
    documents: list[Document]
    queries: list[Query]


def _load_rows(path: Path) -> list[JsonRow]:
    rows: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path.name}: expected a list of objects")
    if any(not all(isinstance(key, str) for key in row) for row in rows):
        raise ValueError(f"{path.name}: object keys must be strings")
    return rows


def load_fixtures(directory: Path = DIRECTORY) -> Fixtures:
    documents: list[Document] = []
    for row in _load_rows(directory / "documents.json"):
        doc_id, text = row.get("id"), row.get("text")
        if not isinstance(doc_id, str) or not isinstance(text, str):
            raise ValueError("documents.json: id and text must be strings")
        documents.append({"id": doc_id, "text": text})

    queries: list[Query] = []
    for row in _load_rows(directory / "qrels.json"):
        query_id, query = row.get("id"), row.get("query")
        relevant, reason = row.get("relevant"), row.get("reason")
        if (
            not isinstance(query_id, str)
            or not isinstance(query, str)
            or not isinstance(relevant, list)
            or not all(isinstance(doc_id, str) for doc_id in relevant)
            or not isinstance(reason, str)
        ):
            raise ValueError("qrels.json: invalid query row")
        queries.append(
            {"id": query_id, "query": query, "relevant": relevant, "reason": reason}
        )
    return Fixtures(documents=documents, queries=queries)
