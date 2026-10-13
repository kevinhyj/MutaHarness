"""Blind equal-weight rank fusion. No labels, model calls or fitted weights."""

from collections.abc import Mapping, Sequence
from itertools import groupby
from math import isfinite
from statistics import mean, pstdev
from typing import TypedDict


class RankedCandidate(TypedDict):
    id: str
    score: float
    program_scores: list[float]
    disagreement: float


def validate_ids(ids: Sequence[str]) -> None:
    if (
        not ids
        or any(not isinstance(id_, str) or not id_ for id_ in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("Query IDs must be nonempty, unique strings")


def query_ids(case: Mapping) -> list[str]:
    queries = case["query"]
    ids = [query["id"] for query in queries]
    validate_ids(ids)
    if set(ids) & {row["id"] for row in case.get("support", [])}:
        raise ValueError("Measured support and unmeasured query IDs must be disjoint")
    if any("fitness" in query for query in queries):
        raise ValueError("Unmeasured queries must not contain fitness labels")
    return ids


def validate_scores(
    scores: Mapping[str, float], ids: Sequence[str], *, bounded: bool = False
) -> None:
    if set(scores) != set(ids):
        raise ValueError("Scores must cover exactly the query IDs")
    for value in scores.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise ValueError("Scores must be finite numbers")
        if bounded and not -1 <= value <= 1:
            raise ValueError("Predictor scores must be in [-1, 1]")


def percentile_ranks(values: Sequence[float]) -> list[float]:
    """Return (average rank - 0.5) / n; tied scores share their midpoint."""
    if not values:
        raise ValueError("Cannot rank an empty sequence")
    ranks = [0.0] * len(values)
    offset = 0
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    for _, tied in groupby(ordered, key=lambda item: item[1]):
        group = list(tied)
        midpoint = (offset + len(group) / 2) / len(values)
        for index, _ in group:
            ranks[index] = midpoint
        offset += len(group)
    return ranks


def rank_ensemble(
    ids: Sequence[str], predictions: Sequence[Mapping[str, float]]
) -> list[RankedCandidate]:
    """Fuse exactly three complete predictions; retain their disagreement for review."""
    validate_ids(ids)
    if len(predictions) != 3:
        raise ValueError("Exactly three predictions are required")
    for scores in predictions:
        validate_scores(scores, ids)
    votes = [percentile_ranks([scores[id_] for id_ in ids]) for scores in predictions]
    return [
        {
            "id": id_,
            "score": 2 * mean(ranks) - 1,
            "program_scores": [scores[id_] for scores in predictions],
            "disagreement": pstdev(ranks),
        }
        for id_, ranks in zip(ids, zip(*votes), strict=True)
    ]
