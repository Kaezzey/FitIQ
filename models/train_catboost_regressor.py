from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))

import ranking_metrics


DEFAULT_DATASET_PATH = Path("data/processed/ranking_eval_dataset.parquet")
DEFAULT_MODEL_OUTPUT_PATH = Path("models/artifacts/first_catboost_regressor.cbm")
DEFAULT_PREDICTIONS_OUTPUT_PATH = Path(
    "models/artifacts/first_catboost_regressor_predictions.parquet"
)
DEFAULT_RESULTS_OUTPUT_PATH = Path("evaluation/results/catboost_regressor_results.json")
DEFAULT_RANDOM_SEED = 20260401
DEFAULT_TOPS_MIN_ROWS = 25
TARGET_COLUMN = "target_score"
GROUP_COLUMN = "query_group_v2"
RANKING_GROUP_COLUMN = "ranking_group"
PREDICTION_COLUMN = "prediction_catboost"
FEATURE_COLUMNS = [
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
    "ranking_group",
]
CATEGORICAL_FEATURE_COLUMNS = ["ranking_group"]
BASELINE_COLUMNS = [
    "mean_rating_score",
    "fit_score",
    "review_volume_score",
    "blended_score",
]
NUMERIC_FEATURE_COLUMNS = [
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
]
EXCLUDED_FROM_TRAINING = {
    "asin",
    "parent_asin",
    "category",
    "subcategory",
    "history_end_timestamp",
    "label_start_exclusive_timestamp",
    "label_end_timestamp",
    "future_mean_rating",
    "future_fit_complaint_rate",
    "future_review_count",
    TARGET_COLUMN,
}
REQUIRED_COLUMNS = {
    "split",
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
    TARGET_COLUMN,
}
CORE_TOPS_SUBCATEGORIES = ("t_shirt", "shirt_top", "sweater", "unknown")
CATBOOST_PARAMS = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 5.0,
    "random_seed": DEFAULT_RANDOM_SEED,
    "od_type": "Iter",
    "od_wait": 50,
    "verbose": 100,
    "allow_writing_files": False,
}


@dataclass(frozen=True)
class ScoreResult:
    name: str
    overall_metrics: dict[str, float]
    per_query_group_metrics: dict[str, dict[str, float]]
    tops_diagnostics: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the first CatBoost regressor for FitIQ."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to data/processed/ranking_eval_dataset.parquet.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=DEFAULT_MODEL_OUTPUT_PATH,
        help="Destination path for the trained CatBoost model.",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=DEFAULT_PREDICTIONS_OUTPUT_PATH,
        help="Destination parquet path for split-level model predictions.",
    )
    parser.add_argument(
        "--results-output",
        type=Path,
        default=DEFAULT_RESULTS_OUTPUT_PATH,
        help="Destination JSON path for model evaluation results.",
    )
    parser.add_argument(
        "--tops-min-subcategory-rows",
        type=int,
        default=DEFAULT_TOPS_MIN_ROWS,
        help="Minimum rows required before emitting tops fine-label metric slices.",
    )
    return parser.parse_args()


def normalize_path(path: Path) -> Path:
    return path.resolve()


def validate_required_columns(frame, required_columns: set[str]) -> None:
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"Ranking evaluation dataset is missing columns: {', '.join(missing)}")


def validate_feature_contract(frame) -> None:
    feature_set = set(FEATURE_COLUMNS)
    if feature_set & EXCLUDED_FROM_TRAINING:
        invalid = sorted(feature_set & EXCLUDED_FROM_TRAINING)
        raise ValueError(f"Training feature list contains excluded columns: {', '.join(invalid)}")

    future_columns = sorted(column for column in FEATURE_COLUMNS if column.startswith("future_"))
    if future_columns:
        raise ValueError(f"Training feature list must not contain future_* columns: {future_columns}")

    missing_features = sorted(feature_set - set(frame.columns))
    if missing_features:
        raise ValueError(f"Training feature columns are missing from dataset: {missing_features}")


def stable_random_score(split: object, asin: object, seed: int) -> float:
    payload = f"{seed}|{split}|{asin}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    integer = int.from_bytes(digest, byteorder="big", signed=False)
    return integer / float((1 << 64) - 1)


def compute_baseline_scores(frame, seed: int):
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
        scored["review_count"].astype(float)
        / (scored["review_count"].astype(float) + 5.0)
    ).clip(lower=0.0, upper=1.0)
    scored["blended_score"] = (
        0.45 * scored["mean_rating_score"]
        + 0.45 * scored["fit_score"]
        + 0.10 * scored["review_volume_score"]
    ).astype(float)
    scored["random_score"] = [
        stable_random_score(split=split, asin=asin, seed=seed)
        for split, asin in zip(scored["split"], scored["asin"], strict=True)
    ]
    return scored


def compute_rmse(frame, score_col: str) -> float:
    errors = frame[score_col].astype(float) - frame[TARGET_COLUMN].astype(float)
    return float(math.sqrt(float((errors**2).mean())))


def compute_overall_metrics(frame, score_col: str) -> dict[str, float]:
    return {
        "rmse": compute_rmse(frame, score_col=score_col),
        "ndcg@10": float(
            ranking_metrics.grouped_ndcg_at_k(
                frame,
                group_col=GROUP_COLUMN,
                truth_col=TARGET_COLUMN,
                score_col=score_col,
                k=10,
            )
        ),
        "ndcg@20": float(
            ranking_metrics.grouped_ndcg_at_k(
                frame,
                group_col=GROUP_COLUMN,
                truth_col=TARGET_COLUMN,
                score_col=score_col,
                k=20,
            )
        ),
        "spearman": float(
            ranking_metrics.grouped_spearman(
                frame,
                group_col=GROUP_COLUMN,
                truth_col=TARGET_COLUMN,
                score_col=score_col,
            )
        ),
    }


def compute_per_ranking_group_metrics(frame, score_col: str) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for ranking_group, group_frame in frame.groupby(GROUP_COLUMN, sort=True):
        truth = group_frame[TARGET_COLUMN].astype(float).tolist()
        scores = group_frame[score_col].astype(float).tolist()
        metrics[str(ranking_group)] = {
            "row_count": int(len(group_frame)),
            "unique_asins": int(group_frame["asin"].nunique()),
            "rmse": compute_rmse(group_frame, score_col=score_col),
            "ndcg@10": float(ranking_metrics.ndcg_at_k(truth, scores, k=10)),
            "ndcg@20": float(ranking_metrics.ndcg_at_k(truth, scores, k=20)),
            "spearman": float(ranking_metrics.spearman_rank_correlation(truth, scores)),
        }
    return metrics


def compute_tops_diagnostics(frame, score_col: str, min_subcategory_rows: int) -> dict[str, object]:
    tops = frame.loc[frame[RANKING_GROUP_COLUMN] == "tops"].copy()
    fine_counts = {key: 0 for key in CORE_TOPS_SUBCATEGORIES}
    metric_slices: dict[str, dict[str, float]] = {}
    skipped_metric_slices: dict[str, int] = {}

    if tops.empty:
        return {
            "row_count": 0,
            "fine_subcategory_counts": fine_counts,
            "fine_subcategory_metric_slices": metric_slices,
            "skipped_metric_slices": skipped_metric_slices,
        }

    observed_counts = tops["subcategory"].value_counts()
    for subcategory, count in observed_counts.items():
        fine_counts[str(subcategory)] = int(count)

    for subcategory, count in fine_counts.items():
        if count <= 0:
            continue
        if count < min_subcategory_rows:
            skipped_metric_slices[subcategory] = int(count)
            continue
        sub_frame = tops.loc[tops["subcategory"] == subcategory]
        truth = sub_frame[TARGET_COLUMN].astype(float).tolist()
        scores = sub_frame[score_col].astype(float).tolist()
        metric_slices[subcategory] = {
            "row_count": int(len(sub_frame)),
            "rmse": compute_rmse(sub_frame, score_col=score_col),
            "ndcg@10": float(ranking_metrics.ndcg_at_k(truth, scores, k=10)),
            "ndcg@20": float(ranking_metrics.ndcg_at_k(truth, scores, k=20)),
            "spearman": float(ranking_metrics.spearman_rank_correlation(truth, scores)),
        }

    return {
        "row_count": int(len(tops)),
        "fine_subcategory_counts": fine_counts,
        "fine_subcategory_metric_slices": metric_slices,
        "skipped_metric_slices": skipped_metric_slices,
    }


def evaluate_score_column(frame, score_col: str, min_subcategory_rows: int) -> ScoreResult:
    return ScoreResult(
        name=score_col,
        overall_metrics=compute_overall_metrics(frame, score_col=score_col),
        per_query_group_metrics=compute_per_ranking_group_metrics(frame, score_col=score_col),
        tops_diagnostics=compute_tops_diagnostics(
            frame,
            score_col=score_col,
            min_subcategory_rows=min_subcategory_rows,
        ),
    )


def choose_best_baseline(results: dict[str, ScoreResult]) -> str:
    return max(
        BASELINE_COLUMNS,
        key=lambda baseline: results[baseline].overall_metrics["ndcg@10"],
    )


def build_metric_deltas(
    model_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
) -> dict[str, float]:
    return {
        "rmse": float(model_metrics["rmse"] - baseline_metrics["rmse"]),
        "ndcg@10": float(model_metrics["ndcg@10"] - baseline_metrics["ndcg@10"]),
        "ndcg@20": float(model_metrics["ndcg@20"] - baseline_metrics["ndcg@20"]),
        "spearman": float(model_metrics["spearman"] - baseline_metrics["spearman"]),
    }


def to_builtin(value):
    if isinstance(value, dict):
        return {str(key): to_builtin(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def build_summary_rows(split_results: dict[str, dict[str, ScoreResult]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    metric_names = ("rmse", "ndcg@10", "ndcg@20", "spearman")
    scorer_names = [PREDICTION_COLUMN, *BASELINE_COLUMNS]
    for split_name, results in split_results.items():
        for scorer_name in scorer_names:
            for metric_name in metric_names:
                rows.append(
                    {
                        "split": split_name,
                        "scorer": scorer_name,
                        "metric": metric_name,
                        "overall_score": float(results[scorer_name].overall_metrics[metric_name]),
                    }
                )
    return rows


def serialize_results(
    dataset_path: Path,
    model_output_path: Path,
    predictions_output_path: Path,
    tops_min_subcategory_rows: int,
    scored_frame,
    split_results: dict[str, dict[str, ScoreResult]],
    training_summary: dict[str, object],
) -> dict[str, object]:
    split_row_counts = {
        split_name: int((scored_frame["split"] == split_name).sum())
        for split_name in ("train", "validation", "test")
    }
    split_group_counts = {
        split_name: {
            str(group): int(count)
            for group, count in scored_frame.loc[
                scored_frame["split"] == split_name,
                GROUP_COLUMN,
            ]
            .value_counts()
            .sort_index()
            .items()
        }
        for split_name in ("train", "validation", "test")
    }
    split_ranking_group_counts = {
        split_name: {
            str(group): int(count)
            for group, count in scored_frame.loc[
                scored_frame["split"] == split_name,
                RANKING_GROUP_COLUMN,
            ]
            .value_counts()
            .sort_index()
            .items()
        }
        for split_name in ("train", "validation", "test")
    }
    split_subcategory_counts = {
        split_name: {
            str(subcategory): int(count)
            for subcategory, count in scored_frame.loc[
                scored_frame["split"] == split_name,
                "subcategory",
            ]
            .value_counts()
            .sort_index()
            .items()
        }
        for split_name in ("train", "validation", "test")
    }

    serialized_splits: dict[str, object] = {}
    for split_name, results in split_results.items():
        best_baseline_name = choose_best_baseline(results)
        model_metrics = results[PREDICTION_COLUMN].overall_metrics
        blended_metrics = results["blended_score"].overall_metrics
        best_baseline_metrics = results[best_baseline_name].overall_metrics
        serialized_splits[split_name] = {
            "row_count": split_row_counts[split_name],
            "unique_asins": int(
                scored_frame.loc[scored_frame["split"] == split_name, "asin"].nunique()
            ),
            "query_group_counts": split_group_counts[split_name],
            "ranking_group_counts": split_ranking_group_counts[split_name],
            "subcategory_counts": split_subcategory_counts[split_name],
            "model": {
                "overall_metrics": model_metrics,
                "per_query_group_metrics": results[PREDICTION_COLUMN].per_query_group_metrics,
                "tops_diagnostics": results[PREDICTION_COLUMN].tops_diagnostics,
            },
            "baselines": {
                baseline_name: {
                    "overall_metrics": result.overall_metrics,
                    "per_query_group_metrics": result.per_query_group_metrics,
                    "tops_diagnostics": result.tops_diagnostics,
                }
                for baseline_name, result in results.items()
                if baseline_name in BASELINE_COLUMNS
            },
            "comparison": {
                "primary_metric": "ndcg@10",
                "best_baseline_name": best_baseline_name,
                "best_baseline_overall_metrics": best_baseline_metrics,
                "delta_vs_blended": build_metric_deltas(model_metrics, blended_metrics),
                "delta_vs_best_baseline": build_metric_deltas(
                    model_metrics, best_baseline_metrics
                ),
                "deltas_vs_baselines": {
                    baseline_name: build_metric_deltas(
                        model_metrics,
                        results[baseline_name].overall_metrics,
                    )
                    for baseline_name in BASELINE_COLUMNS
                },
            },
        }

    return {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "dataset_path": str(dataset_path),
        "model_path": str(model_output_path),
        "predictions_path": str(predictions_output_path),
        "model_type": "CatBoostRegressor",
        "target_column": TARGET_COLUMN,
        "query_group_column": GROUP_COLUMN,
        "ranking_group_column": RANKING_GROUP_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_feature_columns": CATEGORICAL_FEATURE_COLUMNS,
        "tops_min_subcategory_rows": int(tops_min_subcategory_rows),
        "catboost_params": CATBOOST_PARAMS,
        "training_summary": to_builtin(training_summary),
        "dataset_summary": {
            "total_rows": int(len(scored_frame)),
            "unique_asins": int(scored_frame["asin"].nunique()),
            "query_groups": sorted(
                scored_frame[GROUP_COLUMN].dropna().astype(str).unique().tolist()
            ),
            "ranking_groups": sorted(
                scored_frame[RANKING_GROUP_COLUMN].dropna().astype(str).unique().tolist()
            ),
            "splits": split_row_counts,
        },
        "summary_rows": build_summary_rows(split_results),
        "splits": serialized_splits,
    }


def print_split_summary(split_name: str, results: dict[str, ScoreResult]) -> None:
    model_result = results[PREDICTION_COLUMN]
    best_baseline_name = choose_best_baseline(results)
    best_baseline_result = results[best_baseline_name]
    delta_vs_blended = build_metric_deltas(
        model_result.overall_metrics,
        results["blended_score"].overall_metrics,
    )

    print()
    print(f"{split_name.title()} split")
    print(
        "  catboost: "
        f"rmse={model_result.overall_metrics['rmse']:.4f}, "
        f"ndcg@10={model_result.overall_metrics['ndcg@10']:.4f}, "
        f"ndcg@20={model_result.overall_metrics['ndcg@20']:.4f}, "
        f"spearman={model_result.overall_metrics['spearman']:.4f}"
    )
    print(
        f"  best_baseline ({best_baseline_name}): "
        f"rmse={best_baseline_result.overall_metrics['rmse']:.4f}, "
        f"ndcg@10={best_baseline_result.overall_metrics['ndcg@10']:.4f}, "
        f"ndcg@20={best_baseline_result.overall_metrics['ndcg@20']:.4f}, "
        f"spearman={best_baseline_result.overall_metrics['spearman']:.4f}"
    )
    print(
        "  delta_vs_blended: "
        f"rmse={delta_vs_blended['rmse']:.4f}, "
        f"ndcg@10={delta_vs_blended['ndcg@10']:.4f}, "
        f"ndcg@20={delta_vs_blended['ndcg@20']:.4f}, "
        f"spearman={delta_vs_blended['spearman']:.4f}"
    )

    print("  per_query_group_metrics:")
    for query_group, metrics in model_result.per_query_group_metrics.items():
        print(
            f"    {query_group}: "
            f"rmse={metrics['rmse']:.4f}, "
            f"ndcg@10={metrics['ndcg@10']:.4f}, "
            f"ndcg@20={metrics['ndcg@20']:.4f}, "
            f"spearman={metrics['spearman']:.4f}"
        )

    print("  tops_fine_label_counts:")
    for subcategory in CORE_TOPS_SUBCATEGORIES:
        count = int(model_result.tops_diagnostics["fine_subcategory_counts"].get(subcategory, 0))
        print(f"    {subcategory}: {count}")


def prepare_features(frame):
    features = frame.loc[:, FEATURE_COLUMNS].copy()
    for column in NUMERIC_FEATURE_COLUMNS:
        features[column] = features[column].astype(float)
    for column in CATEGORICAL_FEATURE_COLUMNS:
        features[column] = features[column].astype(str)
    return features


def main() -> int:
    args = parse_args()
    if args.tops_min_subcategory_rows < 1:
        print("--tops-min-subcategory-rows must be at least 1.", file=sys.stderr)
        return 1

    dataset_path = normalize_path(args.dataset_path)
    model_output_path = normalize_path(args.model_output)
    predictions_output_path = normalize_path(args.predictions_output)
    results_output_path = normalize_path(args.results_output)

    if not dataset_path.exists():
        print(f"Dataset file does not exist: {dataset_path}", file=sys.stderr)
        return 1

    try:
        import pandas as pd
    except ImportError as exc:
        print(
            "pandas is required to train the first CatBoost scorer. Install it in the "
            "project Python environment, then rerun the script.\n"
            f"Import error: {exc}",
            file=sys.stderr,
        )
        return 1

    try:
        from catboost import CatBoostRegressor, Pool
    except ImportError as exc:
        print(
            "catboost is required to train the first learned scorer. Install it in the "
            "project Python environment, for example with `pip install catboost`, "
            "then rerun the script.\n"
            f"Import error: {exc}",
            file=sys.stderr,
        )
        return 1

    try:
        print("Loading ranking evaluation dataset...")
        frame = pd.read_parquet(dataset_path)
    except Exception as exc:
        print(f"Failed to read ranking evaluation dataset: {exc}", file=sys.stderr)
        return 1

    validate_required_columns(frame, REQUIRED_COLUMNS)
    validate_feature_contract(frame)
    if frame.empty:
        raise ValueError("Ranking evaluation dataset is empty.")
    if frame.duplicated(subset=["split", "asin"]).any():
        raise ValueError("Ranking evaluation dataset contains duplicate (split, asin) rows.")

    print("Computing comparison baselines...")
    frame = compute_baseline_scores(frame, seed=DEFAULT_RANDOM_SEED)

    train_frame = frame.loc[frame["split"] == "train"].copy()
    validation_frame = frame.loc[frame["split"] == "validation"].copy()
    test_frame = frame.loc[frame["split"] == "test"].copy()
    if train_frame.empty or validation_frame.empty or test_frame.empty:
        raise ValueError("Train, validation, and test splits must all be present and non-empty.")

    train_features = prepare_features(train_frame)
    validation_features = prepare_features(validation_frame)
    test_features = prepare_features(test_frame)
    all_features = prepare_features(frame)

    for name, features in (
        ("train", train_features),
        ("validation", validation_features),
        ("test", test_features),
    ):
        if features.isna().any().any():
            raise ValueError(f"{name} features contain missing values.")

    train_target = train_frame[TARGET_COLUMN].astype(float)
    validation_target = validation_frame[TARGET_COLUMN].astype(float)

    print("Training CatBoost regressor...")
    model = CatBoostRegressor(**CATBOOST_PARAMS)
    train_pool = Pool(
        data=train_features,
        label=train_target,
        cat_features=CATEGORICAL_FEATURE_COLUMNS,
    )
    validation_pool = Pool(
        data=validation_features,
        label=validation_target,
        cat_features=CATEGORICAL_FEATURE_COLUMNS,
    )
    model.fit(train_pool, eval_set=validation_pool, use_best_model=True)

    print("Generating predictions...")
    frame[PREDICTION_COLUMN] = model.predict(all_features).astype(float)
    if frame[PREDICTION_COLUMN].isna().any():
        raise ValueError("CatBoost predictions contain missing values.")

    split_results: dict[str, dict[str, ScoreResult]] = {}
    for split_name in ("train", "validation", "test"):
        split_frame = frame.loc[frame["split"] == split_name].copy()
        split_results[split_name] = {
            PREDICTION_COLUMN: evaluate_score_column(
                split_frame,
                score_col=PREDICTION_COLUMN,
                min_subcategory_rows=args.tops_min_subcategory_rows,
            ),
        }
        for baseline_name in BASELINE_COLUMNS:
            split_results[split_name][baseline_name] = evaluate_score_column(
                split_frame,
                score_col=baseline_name,
                min_subcategory_rows=args.tops_min_subcategory_rows,
            )

    predictions_columns = [
        "split",
        "asin",
        "parent_asin",
        "category",
        "query_group_v2",
        "ranking_group",
        "subcategory",
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
        TARGET_COLUMN,
        PREDICTION_COLUMN,
        *BASELINE_COLUMNS,
    ]
    predictions_frame = frame.loc[:, predictions_columns].copy()

    split_order = {"train": 0, "validation": 1, "test": 2}
    predictions_frame["split_order"] = predictions_frame["split"].map(split_order)
    predictions_frame = predictions_frame.sort_values(
        ["split_order", "query_group_v2", "ranking_group", "subcategory", "asin"],
        kind="stable",
    ).drop(columns=["split_order"])
    predictions_frame = predictions_frame.reset_index(drop=True)

    if len(predictions_frame) != len(frame):
        raise ValueError("Prediction table row count does not match the ranking dataset.")

    training_summary = {
        "best_iteration": int(model.get_best_iteration()),
        "best_score": to_builtin(model.get_best_score()),
        "feature_importance": {
            name: float(value)
            for name, value in zip(
                FEATURE_COLUMNS,
                model.get_feature_importance(),
                strict=True,
            )
        },
    }

    print("Writing model and evaluation artifacts...")
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_output_path))

    predictions_output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_frame.to_parquet(predictions_output_path, index=False)

    results_output_path.parent.mkdir(parents=True, exist_ok=True)
    results_payload = serialize_results(
        dataset_path=dataset_path,
        model_output_path=model_output_path,
        predictions_output_path=predictions_output_path,
        tops_min_subcategory_rows=args.tops_min_subcategory_rows,
        scored_frame=predictions_frame,
        split_results=split_results,
        training_summary=training_summary,
    )
    results_output_path.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print()
    print("CatBoost regressor training complete")
    print(f"Dataset path: {dataset_path}")
    print(f"Model path: {model_output_path}")
    print(f"Predictions path: {predictions_output_path}")
    print(f"Results path: {results_output_path}")
    print(
        f"Best iteration: {training_summary['best_iteration']} "
        f"(validation RMSE={split_results['validation'][PREDICTION_COLUMN].overall_metrics['rmse']:.4f})"
    )
    print_split_summary("validation", split_results["validation"])
    print_split_summary("test", split_results["test"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
