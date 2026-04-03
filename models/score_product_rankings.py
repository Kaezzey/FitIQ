from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from train_catboost_regressor import (
    CATEGORICAL_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    normalize_path,
    prepare_features,
)
from train_query_group_hybrid_ranker import add_rank_score_columns


DEFAULT_FEATURES_PATH = Path("data/processed/product_features.parquet")
DEFAULT_MODEL_PATH = Path("models/artifacts/tuned_catboost_ranker.cbm")
DEFAULT_POLICY_PATH = Path("models/artifacts/query_group_hybrid_policy.json")
DEFAULT_OUTPUT_PATH = Path("data/processed/product_rankings_v1.parquet")
DEFAULT_SUMMARY_OUTPUT_PATH = Path("data/processed/product_rankings_v1_summary.json")
UNKNOWN_QUERY_GROUP = "unknown"
PREDICTION_COLUMN = "prediction_catboost_ranker_tuned"
FINAL_SCORE_COLUMN = "score_fitiq_v1"
REQUIRED_COLUMNS = {
    "asin",
    "parent_asin",
    "category",
    "subcategory",
    "ranking_group",
    "query_group_v2",
    "is_apparel",
    *FEATURE_COLUMNS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score and rank full-corpus products with the tuned CatBoost ranker "
            "and learned query-group hybrid policy."
        )
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=DEFAULT_FEATURES_PATH,
        help="Path to data/processed/product_features.parquet.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the tuned CatBoost ranker model.",
    )
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=DEFAULT_POLICY_PATH,
        help="Path to the learned query-group hybrid policy JSON.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination parquet path for scored product rankings.",
    )
    parser.add_argument(
        "--summary-output-path",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT_PATH,
        help="Destination JSON path for a scoring summary.",
    )
    return parser.parse_args()


def validate_required_columns(frame, required_columns: set[str], name: str) -> None:
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def compute_serving_baseline_scores(frame):
    scored = frame.copy()
    scored["mean_rating_score"] = ((scored["mean_rating"].astype(float) - 1.0) / 4.0).clip(
        lower=0.0,
        upper=1.0,
    )
    scored["fit_score"] = (1.0 - scored["fit_complaint_rate"].astype(float)).clip(
        lower=0.0,
        upper=1.0,
    )
    scored["review_volume_score"] = (
        scored["review_count"].astype(float) / (scored["review_count"].astype(float) + 5.0)
    ).clip(lower=0.0, upper=1.0)
    scored["blended_score"] = (
        0.45 * scored["mean_rating_score"]
        + 0.45 * scored["fit_score"]
        + 0.10 * scored["review_volume_score"]
    ).astype(float)
    return scored


def load_policy(path: Path) -> dict[str, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Policy file must contain a JSON object: {path}")
    return payload


def apply_hybrid_policy(frame, policy: dict[str, dict[str, object]]):
    scored = add_rank_score_columns(frame, (PREDICTION_COLUMN, "blended_score"))
    model_rank_column = f"{PREDICTION_COLUMN}__rank_score"
    baseline_rank_column = "blended_score__rank_score"
    baseline_weights = []
    model_weights = []
    selection_sources = []
    for query_group in scored["query_group_v2"]:
        group_policy = policy.get(str(query_group))
        if group_policy is None:
            baseline_weights.append(1.0)
            model_weights.append(0.0)
            selection_sources.append("missing_policy_fallback")
            continue
        baseline_weights.append(float(group_policy.get("baseline_weight", 1.0)))
        model_weights.append(float(group_policy.get("model_weight", 0.0)))
        selection_sources.append("policy")

    scored["selected_baseline_weight"] = baseline_weights
    scored["selected_model_weight"] = model_weights
    scored["policy_source"] = selection_sources
    scored[FINAL_SCORE_COLUMN] = (
        scored["selected_baseline_weight"] * scored[baseline_rank_column].astype(float)
        + scored["selected_model_weight"] * scored[model_rank_column].astype(float)
    ).astype(float)
    return scored.drop(columns=[model_rank_column, baseline_rank_column])


def rank_within_query_groups(frame):
    ranked = frame.copy()
    ranked["query_group_size"] = ranked.groupby("query_group_v2")["asin"].transform("size")
    ranked = ranked.sort_values(
        ["query_group_v2", FINAL_SCORE_COLUMN, "review_count", "mean_rating", "asin"],
        ascending=[True, False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    ranked["rank_in_query_group"] = ranked.groupby("query_group_v2").cumcount() + 1
    ranked["rank_percentile"] = 1.0
    mask = ranked["query_group_size"] > 1
    ranked.loc[mask, "rank_percentile"] = 1.0 - (
        (ranked.loc[mask, "rank_in_query_group"].astype(float) - 1.0)
        / (ranked.loc[mask, "query_group_size"].astype(float) - 1.0)
    )
    ranked["rank_percentile"] = ranked["rank_percentile"].astype(float)
    return ranked


def build_summary(
    features_path: Path,
    model_path: Path,
    policy_path: Path,
    output_path: Path,
    ranked,
) -> dict[str, object]:
    query_group_counts = Counter(ranked["query_group_v2"].tolist())
    ranking_group_counts = Counter(ranked["ranking_group"].tolist())
    policy_source_counts = Counter(ranked["policy_source"].tolist())
    top_examples: dict[str, list[dict[str, object]]] = {}
    for query_group, group_frame in ranked.groupby("query_group_v2", sort=True):
        top_rows = group_frame.head(5)
        top_examples[str(query_group)] = [
            {
                "asin": str(row.asin),
                "subcategory": str(row.subcategory),
                "review_count": int(row.review_count),
                "mean_rating": float(row.mean_rating),
                "score_fitiq_v1": float(getattr(row, FINAL_SCORE_COLUMN)),
                "rank_in_query_group": int(row.rank_in_query_group),
            }
            for row in top_rows.itertuples(index=False)
        ]

    return {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "features_path": str(features_path),
        "model_path": str(model_path),
        "policy_path": str(policy_path),
        "output_path": str(output_path),
        "score_column": FINAL_SCORE_COLUMN,
        "dataset_summary": {
            "total_ranked_products": int(len(ranked)),
            "unique_asins": int(ranked["asin"].nunique()),
            "query_group_counts": {str(key): int(value) for key, value in query_group_counts.items()},
            "ranking_group_counts": {str(key): int(value) for key, value in ranking_group_counts.items()},
            "policy_source_counts": {str(key): int(value) for key, value in policy_source_counts.items()},
        },
        "top_examples_by_query_group": top_examples,
    }


def print_summary(ranked, output_path: Path, summary_output_path: Path) -> None:
    query_group_counts = Counter(ranked["query_group_v2"].tolist())
    ranking_group_counts = Counter(ranked["ranking_group"].tolist())
    policy_source_counts = Counter(ranked["policy_source"].tolist())

    print()
    print("Product ranking build complete")
    print(f"Output path: {output_path}")
    print(f"Summary path: {summary_output_path}")

    print()
    print("Ranking coverage")
    print(f"  ranked_products: {len(ranked)}")
    print(f"  unique_asins: {ranked['asin'].nunique()}")

    print()
    print("Query group counts")
    for query_group, count in query_group_counts.most_common():
        print(f"  {query_group}: {count}")

    print()
    print("Ranking group counts")
    for ranking_group, count in ranking_group_counts.most_common():
        print(f"  {ranking_group}: {count}")

    print()
    print("Policy source counts")
    for source, count in policy_source_counts.most_common():
        print(f"  {source}: {count}")


def main() -> int:
    args = parse_args()
    features_path = normalize_path(args.features_path)
    model_path = normalize_path(args.model_path)
    policy_path = normalize_path(args.policy_path)
    output_path = normalize_path(args.output_path)
    summary_output_path = normalize_path(args.summary_output_path)

    for path in (features_path, model_path, policy_path):
        if not path.exists():
            print(f"Input file does not exist: {path}", file=sys.stderr)
            return 1

    try:
        import pandas as pd
    except ImportError as exc:
        print(
            "pandas is required to score product rankings.\n"
            f"Import error: {exc}",
            file=sys.stderr,
        )
        return 1

    try:
        from catboost import CatBoostRanker, Pool
    except ImportError as exc:
        print(
            "catboost is required to score product rankings.\n"
            f"Import error: {exc}",
            file=sys.stderr,
        )
        return 1

    print("Loading product features and policy...")
    features = pd.read_parquet(features_path)
    validate_required_columns(features, REQUIRED_COLUMNS, str(features_path))
    policy = load_policy(policy_path)

    print("Filtering to rankable apparel products...")
    features = features.loc[
        features["is_apparel"].astype(bool) & (features["query_group_v2"] != UNKNOWN_QUERY_GROUP)
    ].copy()
    if features.empty:
        raise ValueError("No rankable apparel products remain after filtering product features.")

    print("Computing heuristic scores...")
    features = compute_serving_baseline_scores(features)

    print("Scoring tuned CatBoost ranker...")
    model = CatBoostRanker()
    model.load_model(str(model_path))
    model_features = prepare_features(features)
    pool = Pool(data=model_features, cat_features=CATEGORICAL_FEATURE_COLUMNS)
    features[PREDICTION_COLUMN] = model.predict(pool).astype(float)
    if features[PREDICTION_COLUMN].isna().any():
        raise ValueError("CatBoost ranking predictions contain missing values.")

    print("Applying learned query-group hybrid policy...")
    ranked = apply_hybrid_policy(features, policy=policy)
    ranked = rank_within_query_groups(ranked)

    output_columns = [
        "asin",
        "parent_asin",
        "category",
        "subcategory",
        "ranking_group",
        "query_group_v2",
        "mean_rating",
        "rating_variance",
        "review_count",
        "log_review_count",
        "fit_complaint_rate",
        "low_star_rate",
        "high_star_rate",
        "too_small_rate",
        "too_large_rate",
        "tight_rate",
        "loose_rate",
        "quality_complaint_rate",
        "verified_purchase_rate",
        "helpful_votes_mean",
        "helpful_votes_sum_log",
        "recent_review_count",
        "recent_mean_rating",
        "mean_rating_score",
        "fit_score",
        "review_volume_score",
        "blended_score",
        PREDICTION_COLUMN,
        "selected_baseline_weight",
        "selected_model_weight",
        "policy_source",
        FINAL_SCORE_COLUMN,
        "rank_in_query_group",
        "query_group_size",
        "rank_percentile",
    ]
    ranked = ranked.loc[:, output_columns]

    print("Writing ranked product artifacts...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_parquet(output_path, index=False)
    summary_payload = build_summary(
        features_path=features_path,
        model_path=model_path,
        policy_path=policy_path,
        output_path=output_path,
        ranked=ranked,
    )
    summary_output_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print_summary(ranked=ranked, output_path=output_path, summary_output_path=summary_output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
