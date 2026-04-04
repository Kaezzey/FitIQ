from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RANKINGS_PATH = PROJECT_ROOT / "data" / "processed" / "product_rankings_v1.parquet"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "data" / "processed" / "product_rankings_v1_summary.json"
SERVING_COLUMNS = [
    "asin",
    "parent_asin",
    "category",
    "subcategory",
    "ranking_group",
    "query_group_v2",
    "score_fitiq_v1",
    "rank_in_query_group",
    "query_group_size",
    "rank_percentile",
    "review_count",
    "mean_rating",
]


class RankingStoreError(RuntimeError):
    """Raised when the ranking artifact cannot be loaded or validated."""


def _to_builtin(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _record_from_row(row: Any) -> dict[str, Any]:
    return {
        "asin": str(row.asin),
        "parent_asin": None if row.parent_asin is None else str(row.parent_asin),
        "category": None if row.category is None else str(row.category),
        "subcategory": str(row.subcategory),
        "ranking_group": str(row.ranking_group),
        "query_group_v2": str(row.query_group_v2),
        "score_fitiq_v1": float(row.score_fitiq_v1),
        "rank_in_query_group": int(row.rank_in_query_group),
        "query_group_size": int(row.query_group_size),
        "rank_percentile": float(row.rank_percentile),
        "review_count": int(row.review_count),
        "mean_rating": float(row.mean_rating),
    }


@dataclass(frozen=True)
class RankingStore:
    rankings_path: Path
    summary_path: Path
    loaded_at_utc: str
    artifact_version: str
    artifact_generated_at_utc: str | None
    score_column: str | None
    query_group_column: str
    total_ranked_products: int
    query_group_count: int
    query_group_counts: dict[str, int]
    _rankings: Any
    _query_group_ranges: dict[str, tuple[int, int]]
    _asin_to_row_index: dict[str, int]

    def health_payload(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "artifact_version": self.artifact_version,
            "artifact_generated_at_utc": self.artifact_generated_at_utc,
            "artifact_loaded_at_utc": self.loaded_at_utc,
            "score_column": self.score_column,
            "query_group_column": self.query_group_column,
            "total_ranked_products": self.total_ranked_products,
            "query_group_count": self.query_group_count,
        }

    def list_query_groups(self) -> list[dict[str, Any]]:
        return [
            {"query_group_v2": query_group, "product_count": int(count)}
            for query_group, count in sorted(
                self.query_group_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]

    def get_rankings(self, query_group_v2: str, limit: int, offset: int) -> dict[str, Any]:
        query_group = str(query_group_v2)
        range_bounds = self._query_group_ranges.get(query_group)
        if range_bounds is None:
            raise KeyError(query_group)

        start, end = range_bounds
        total_products = end - start
        slice_start = start + offset
        slice_end = min(end, slice_start + limit)
        if slice_start >= end:
            items: list[dict[str, Any]] = []
        else:
            items = [
                _record_from_row(row)
                for row in self._rankings.iloc[slice_start:slice_end].itertuples(index=False)
            ]

        return {
            "query_group_v2": query_group,
            "total_products": int(total_products),
            "limit": int(limit),
            "offset": int(offset),
            "items": items,
        }

    def get_product(self, asin: str) -> dict[str, Any]:
        row_index = self._asin_to_row_index.get(str(asin))
        if row_index is None:
            raise KeyError(asin)
        row = self._rankings.iloc[row_index]
        return {column: _to_builtin(row[column]) for column in SERVING_COLUMNS}


def _validate_required_columns(frame, required_columns: list[str], name: str) -> None:
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise RankingStoreError(f"{name} is missing required columns: {', '.join(missing)}")


def build_ranking_store(
    rankings_path: Path = DEFAULT_RANKINGS_PATH,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
) -> RankingStore:
    rankings_path = rankings_path.resolve()
    summary_path = summary_path.resolve()

    if not rankings_path.exists():
        raise RankingStoreError(f"Ranking parquet does not exist: {rankings_path}")
    if not summary_path.exists():
        raise RankingStoreError(f"Ranking summary JSON does not exist: {summary_path}")

    try:
        import pandas as pd
    except ImportError as exc:
        raise RankingStoreError(
            "pandas is required to load the ranking artifact for the API."
        ) from exc

    try:
        rankings = pd.read_parquet(rankings_path, columns=SERVING_COLUMNS)
    except Exception as exc:
        raise RankingStoreError(f"Failed to load ranking parquet: {exc}") from exc

    _validate_required_columns(rankings, SERVING_COLUMNS, str(rankings_path))
    if rankings.empty:
        raise RankingStoreError("Ranking parquet is empty.")
    if rankings["asin"].duplicated().any():
        raise RankingStoreError("Ranking parquet contains duplicate ASIN rows.")
    if rankings["query_group_v2"].isna().any():
        raise RankingStoreError("Ranking parquet contains null query_group_v2 values.")
    if rankings["rank_in_query_group"].isna().any():
        raise RankingStoreError("Ranking parquet contains null rank_in_query_group values.")

    rankings = rankings.sort_values(
        ["query_group_v2", "rank_in_query_group", "asin"],
        ascending=[True, True, True],
        kind="stable",
    ).reset_index(drop=True)

    computed_counts = rankings["query_group_v2"].value_counts(sort=False).to_dict()
    query_group_ranges: dict[str, tuple[int, int]] = {}
    start = 0
    for query_group, count in rankings.groupby("query_group_v2", sort=False).size().items():
        count_int = int(count)
        query_group_ranges[str(query_group)] = (start, start + count_int)
        start += count_int

    for query_group, group_frame in rankings.groupby("query_group_v2", sort=False):
        expected_ranks = list(range(1, len(group_frame) + 1))
        observed_ranks = group_frame["rank_in_query_group"].astype(int).tolist()
        if observed_ranks != expected_ranks:
            raise RankingStoreError(
                f"Ranking parquet has non-contiguous rank_in_query_group values for {query_group}."
            )
        stored_sizes = group_frame["query_group_size"].astype(int).unique().tolist()
        if stored_sizes != [len(group_frame)]:
            raise RankingStoreError(
                f"Ranking parquet has inconsistent query_group_size for {query_group}."
            )

    try:
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RankingStoreError(f"Failed to load ranking summary JSON: {exc}") from exc
    if not isinstance(summary_payload, dict):
        raise RankingStoreError("Ranking summary JSON must contain an object payload.")

    summary_counts = summary_payload.get("dataset_summary", {}).get("query_group_counts", {})
    if summary_counts:
        normalized_summary_counts = {str(key): int(value) for key, value in summary_counts.items()}
        normalized_computed_counts = {str(key): int(value) for key, value in computed_counts.items()}
        if normalized_summary_counts != normalized_computed_counts:
            raise RankingStoreError(
                "Ranking summary JSON query_group_counts do not match the parquet artifact."
            )

    asin_to_row_index = {
        str(asin): int(index) for index, asin in enumerate(rankings["asin"].astype(str).tolist())
    }

    return RankingStore(
        rankings_path=rankings_path,
        summary_path=summary_path,
        loaded_at_utc=datetime.now(tz=UTC).isoformat(),
        artifact_version=str(summary_payload.get("artifact_version", "product_rankings_v1")),
        artifact_generated_at_utc=summary_payload.get("generated_at_utc"),
        score_column=summary_payload.get("score_column"),
        query_group_column=str(summary_payload.get("query_group_column", "query_group_v2")),
        total_ranked_products=int(len(rankings)),
        query_group_count=int(rankings["query_group_v2"].nunique()),
        query_group_counts={str(key): int(value) for key, value in computed_counts.items()},
        _rankings=rankings,
        _query_group_ranges=query_group_ranges,
        _asin_to_row_index=asin_to_row_index,
    )
