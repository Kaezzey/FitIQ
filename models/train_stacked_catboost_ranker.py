from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from train_catboost_ranker import sort_for_query_groups
from train_catboost_regressor import (
    BASELINE_COLUMNS,
    CORE_TOPS_SUBCATEGORIES,
    DEFAULT_DATASET_PATH,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TOPS_MIN_ROWS,
    EXCLUDED_FROM_TRAINING,
    GROUP_COLUMN,
    NUMERIC_FEATURE_COLUMNS,
    RANKING_GROUP_COLUMN,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    ScoreResult,
    build_metric_deltas,
    choose_best_baseline,
    compute_baseline_scores,
    evaluate_score_column,
    normalize_path,
    to_builtin,
    validate_required_columns,
)


DEFAULT_MODEL_OUTPUT_PATH = Path("models/artifacts/stacked_catboost_ranker.cbm")
DEFAULT_PREDICTIONS_OUTPUT_PATH = Path(
    "models/artifacts/stacked_catboost_ranker_predictions.parquet"
)
DEFAULT_RESULTS_OUTPUT_PATH = Path("evaluation/results/stacked_catboost_ranker_results.json")
PREDICTION_COLUMN = "prediction_catboost_ranker_stacked"
STACKED_NUMERIC_FEATURE_COLUMNS = [*NUMERIC_FEATURE_COLUMNS, *BASELINE_COLUMNS]
STACKED_CATEGORICAL_FEATURE_COLUMNS = [GROUP_COLUMN, RANKING_GROUP_COLUMN]
STACKED_FEATURE_COLUMNS = [
    *STACKED_NUMERIC_FEATURE_COLUMNS,
    *STACKED_CATEGORICAL_FEATURE_COLUMNS,
]
STACKED_RANKER_PARAMS = {
    "loss_function": "QuerySoftMax",
    "eval_metric": "NDCG:top=10",
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 10.0,
    "random_seed": DEFAULT_RANDOM_SEED,
    "od_type": "Iter",
    "od_wait": 50,
    "verbose": 100,
    "allow_writing_files": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a stacked CatBoost ranker using baseline heuristic scores as features."
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
        help="Destination path for the stacked CatBoost ranker.",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=DEFAULT_PREDICTIONS_OUTPUT_PATH,
        help="Destination parquet path for stacked ranker predictions.",
    )
    parser.add_argument(
        "--results-output",
        type=Path,
        default=DEFAULT_RESULTS_OUTPUT_PATH,
        help="Destination JSON path for stacked ranker results.",
    )
    parser.add_argument(
        "--tops-min-subcategory-rows",
        type=int,
        default=DEFAULT_TOPS_MIN_ROWS,
        help="Minimum rows required before emitting tops fine-label metric slices.",
    )
    return parser.parse_args()


def validate_stacked_feature_contract(frame) -> None:
    feature_set = set(STACKED_FEATURE_COLUMNS)
    invalid = sorted(feature_set & EXCLUDED_FROM_TRAINING)
    if invalid:
        raise ValueError(
            f"Stacked feature list contains excluded columns: {', '.join(invalid)}"
        )

    future_columns = sorted(
        column for column in STACKED_FEATURE_COLUMNS if column.startswith("future_")
    )
    if future_columns:
        raise ValueError(
            f"Stacked feature list must not contain future_* columns: {future_columns}"
        )

    missing = sorted(feature_set - set(frame.columns))
    if missing:
        raise ValueError(
            f"Stacked feature columns are missing from dataset: {', '.join(missing)}"
        )


def prepare_stacked_features(frame):
    features = frame.loc[:, STACKED_FEATURE_COLUMNS].copy()
    for column in STACKED_NUMERIC_FEATURE_COLUMNS:
        features[column] = features[column].astype(float)
    for column in STACKED_CATEGORICAL_FEATURE_COLUMNS:
        features[column] = features[column].astype(str)
    return features


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
    split_query_group_counts = {
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
            "query_group_counts": split_query_group_counts[split_name],
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
                    model_metrics,
                    best_baseline_metrics,
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
        "model_type": "CatBoostRanker",
        "target_column": TARGET_COLUMN,
        "query_group_column": GROUP_COLUMN,
        "ranking_group_column": RANKING_GROUP_COLUMN,
        "feature_columns": STACKED_FEATURE_COLUMNS,
        "numeric_feature_columns": STACKED_NUMERIC_FEATURE_COLUMNS,
        "categorical_feature_columns": STACKED_CATEGORICAL_FEATURE_COLUMNS,
        "tops_min_subcategory_rows": int(tops_min_subcategory_rows),
        "catboost_params": STACKED_RANKER_PARAMS,
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
        "  stacked_ranker: "
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
            "pandas is required to train the stacked CatBoost ranker. Install it in the "
            "project Python environment, then rerun the script.\n"
            f"Import error: {exc}",
            file=sys.stderr,
        )
        return 1

    try:
        from catboost import CatBoostRanker, Pool
    except ImportError as exc:
        print(
            "catboost is required to train the stacked ranking model. Install it in the "
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
    if frame.empty:
        raise ValueError("Ranking evaluation dataset is empty.")
    if frame.duplicated(subset=["split", "asin"]).any():
        raise ValueError("Ranking evaluation dataset contains duplicate (split, asin) rows.")

    print("Computing baseline features for stacking...")
    frame = compute_baseline_scores(frame, seed=DEFAULT_RANDOM_SEED)
    validate_stacked_feature_contract(frame)

    train_frame = sort_for_query_groups(frame.loc[frame["split"] == "train"].copy())
    validation_frame = sort_for_query_groups(frame.loc[frame["split"] == "validation"].copy())
    test_frame = sort_for_query_groups(frame.loc[frame["split"] == "test"].copy())
    if train_frame.empty or validation_frame.empty or test_frame.empty:
        raise ValueError("Train, validation, and test splits must all be present and non-empty.")

    train_features = prepare_stacked_features(train_frame)
    validation_features = prepare_stacked_features(validation_frame)
    all_features = prepare_stacked_features(frame)

    for name, features in (
        ("train", train_features),
        ("validation", validation_features),
        ("all", all_features),
    ):
        if features.isna().any().any():
            raise ValueError(f"{name} stacked features contain missing values.")

    train_target = train_frame[TARGET_COLUMN].astype(float)
    validation_target = validation_frame[TARGET_COLUMN].astype(float)

    print("Training stacked CatBoost ranker...")
    model = CatBoostRanker(**STACKED_RANKER_PARAMS)
    train_pool = Pool(
        data=train_features,
        label=train_target,
        group_id=train_frame["query_id"].astype(str),
        cat_features=STACKED_CATEGORICAL_FEATURE_COLUMNS,
    )
    validation_pool = Pool(
        data=validation_features,
        label=validation_target,
        group_id=validation_frame["query_id"].astype(str),
        cat_features=STACKED_CATEGORICAL_FEATURE_COLUMNS,
    )
    model.fit(train_pool, eval_set=validation_pool, use_best_model=True)

    print("Generating stacked ranker predictions...")
    frame[PREDICTION_COLUMN] = model.predict(all_features).astype(float)
    if frame[PREDICTION_COLUMN].isna().any():
        raise ValueError("Stacked CatBoost ranker predictions contain missing values.")

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
        GROUP_COLUMN,
        RANKING_GROUP_COLUMN,
        "subcategory",
        *STACKED_NUMERIC_FEATURE_COLUMNS,
        TARGET_COLUMN,
        PREDICTION_COLUMN,
    ]
    predictions_frame = frame.loc[:, predictions_columns].copy()
    split_order = {"train": 0, "validation": 1, "test": 2}
    predictions_frame["split_order"] = predictions_frame["split"].map(split_order)
    predictions_frame = predictions_frame.sort_values(
        ["split_order", GROUP_COLUMN, RANKING_GROUP_COLUMN, "subcategory", "asin"],
        kind="stable",
    ).drop(columns=["split_order"])
    predictions_frame = predictions_frame.reset_index(drop=True)

    training_summary = {
        "best_iteration": int(model.get_best_iteration()),
        "best_score": to_builtin(model.get_best_score()),
        "feature_importance": {
            name: float(value)
            for name, value in zip(
                STACKED_FEATURE_COLUMNS,
                model.get_feature_importance(train_pool),
                strict=True,
            )
        },
    }

    print("Writing stacked ranker artifacts...")
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
    print("Stacked CatBoost ranker training complete")
    print(f"Dataset path: {dataset_path}")
    print(f"Model path: {model_output_path}")
    print(f"Predictions path: {predictions_output_path}")
    print(f"Results path: {results_output_path}")
    print(
        f"Best iteration: {training_summary['best_iteration']} "
        f"(validation NDCG@10={split_results['validation'][PREDICTION_COLUMN].overall_metrics['ndcg@10']:.4f})"
    )
    print_split_summary("validation", split_results["validation"])
    print_split_summary("test", split_results["test"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
