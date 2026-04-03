from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from train_catboost_regressor import (
    BASELINE_COLUMNS,
    CORE_TOPS_SUBCATEGORIES,
    DEFAULT_TOPS_MIN_ROWS,
    GROUP_COLUMN,
    RANKING_GROUP_COLUMN,
    TARGET_COLUMN,
    ScoreResult,
    build_metric_deltas,
    choose_best_baseline,
    evaluate_score_column,
    normalize_path,
    to_builtin,
)


DEFAULT_SOURCE_PREDICTIONS_PATH = Path("models/artifacts/tuned_catboost_ranker_predictions.parquet")
DEFAULT_POLICY_OUTPUT_PATH = Path("models/artifacts/query_group_hybrid_policy.json")
DEFAULT_PREDICTIONS_OUTPUT_PATH = Path(
    "models/artifacts/query_group_hybrid_predictions.parquet"
)
DEFAULT_RESULTS_OUTPUT_PATH = Path("evaluation/results/query_group_hybrid_results.json")
DEFAULT_MODEL_SCORE_COLUMN = "prediction_catboost_ranker_tuned"
DEFAULT_BASELINE_SCORE_COLUMN = "blended_score"
PREDICTION_COLUMN = "prediction_query_group_hybrid"
WEIGHT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
REQUIRED_COLUMNS = {
    "split",
    "asin",
    "parent_asin",
    "category",
    "subcategory",
    GROUP_COLUMN,
    RANKING_GROUP_COLUMN,
    TARGET_COLUMN,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a query-group hybrid ranker by blending tuned CatBoost ranker "
            "scores with blended_score using validation-selected weights."
        )
    )
    parser.add_argument(
        "--source-predictions",
        type=Path,
        default=DEFAULT_SOURCE_PREDICTIONS_PATH,
        help="Path to the tuned ranker predictions parquet.",
    )
    parser.add_argument(
        "--policy-output",
        type=Path,
        default=DEFAULT_POLICY_OUTPUT_PATH,
        help="Destination JSON path for the learned query-group weight policy.",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=DEFAULT_PREDICTIONS_OUTPUT_PATH,
        help="Destination parquet path for hybrid predictions.",
    )
    parser.add_argument(
        "--results-output",
        type=Path,
        default=DEFAULT_RESULTS_OUTPUT_PATH,
        help="Destination JSON path for hybrid evaluation results.",
    )
    parser.add_argument(
        "--model-score-column",
        type=str,
        default=DEFAULT_MODEL_SCORE_COLUMN,
        help="Column name for the learned challenger score.",
    )
    parser.add_argument(
        "--baseline-score-column",
        type=str,
        default=DEFAULT_BASELINE_SCORE_COLUMN,
        help="Column name for the heuristic champion score.",
    )
    parser.add_argument(
        "--tops-min-subcategory-rows",
        type=int,
        default=DEFAULT_TOPS_MIN_ROWS,
        help="Minimum rows required before emitting tops fine-label metric slices.",
    )
    return parser.parse_args()


def validate_required_columns(frame, required_columns: set[str]) -> None:
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"Predictions frame is missing required columns: {', '.join(missing)}")


def add_rank_score_columns(frame, score_columns: tuple[str, ...]):
    ranked = frame.copy()
    for score_column in score_columns:
        rank_column = f"{score_column}__rank_score"
        ranks = ranked.groupby(GROUP_COLUMN, sort=False)[score_column].rank(
            method="average",
            ascending=False,
        )
        group_sizes = ranked.groupby(GROUP_COLUMN, sort=False)[score_column].transform("size")
        ranked[rank_column] = 1.0
        mask = group_sizes > 1
        ranked.loc[mask, rank_column] = 1.0 - (
            (ranks.loc[mask] - 1.0) / (group_sizes.loc[mask] - 1.0)
        )
        ranked[rank_column] = ranked[rank_column].astype(float)
    return ranked


def score_group(truth: list[float], scores: list[float]) -> dict[str, float]:
    import ranking_metrics

    return {
        "ndcg@10": float(ranking_metrics.ndcg_at_k(truth, scores, k=10)),
        "ndcg@20": float(ranking_metrics.ndcg_at_k(truth, scores, k=20)),
        "spearman": float(ranking_metrics.spearman_rank_correlation(truth, scores)),
    }


def choose_weight_policy(
    validation_frame,
    model_score_column: str,
    baseline_score_column: str,
) -> dict[str, dict[str, object]]:
    policy: dict[str, dict[str, object]] = {}
    model_rank_column = f"{model_score_column}__rank_score"
    baseline_rank_column = f"{baseline_score_column}__rank_score"

    for query_group, group_frame in validation_frame.groupby(GROUP_COLUMN, sort=True):
        truth = group_frame[TARGET_COLUMN].astype(float).tolist()
        best_payload = None
        best_key = None
        for baseline_weight in WEIGHT_GRID:
            hybrid_scores = (
                baseline_weight * group_frame[baseline_rank_column].astype(float)
                + (1.0 - baseline_weight) * group_frame[model_rank_column].astype(float)
            ).tolist()
            metrics = score_group(truth, hybrid_scores)
            selection_key = (
                metrics["ndcg@10"],
                metrics["ndcg@20"],
                metrics["spearman"],
                baseline_weight,
            )
            if best_key is None or selection_key > best_key:
                best_key = selection_key
                best_payload = {
                    "baseline_weight": float(baseline_weight),
                    "model_weight": float(1.0 - baseline_weight),
                    "validation_metrics": metrics,
                    "row_count": int(len(group_frame)),
                }
        policy[str(query_group)] = best_payload
    return policy


def apply_weight_policy(
    frame,
    policy: dict[str, dict[str, object]],
    model_score_column: str,
    baseline_score_column: str,
):
    scored = add_rank_score_columns(frame, (model_score_column, baseline_score_column))
    model_rank_column = f"{model_score_column}__rank_score"
    baseline_rank_column = f"{baseline_score_column}__rank_score"
    baseline_weights = [
        float(policy.get(str(query_group), {}).get("baseline_weight", 1.0))
        for query_group in scored[GROUP_COLUMN]
    ]
    scored["selected_baseline_weight"] = baseline_weights
    scored["selected_model_weight"] = 1.0 - scored["selected_baseline_weight"]
    scored[PREDICTION_COLUMN] = (
        scored["selected_baseline_weight"] * scored[baseline_rank_column].astype(float)
        + scored["selected_model_weight"] * scored[model_rank_column].astype(float)
    ).astype(float)
    return scored.drop(columns=[model_rank_column, baseline_rank_column])


def build_summary_rows(split_results: dict[str, dict[str, ScoreResult]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    scorer_names = [PREDICTION_COLUMN, DEFAULT_MODEL_SCORE_COLUMN, *BASELINE_COLUMNS]
    metric_names = ("rmse", "ndcg@10", "ndcg@20", "spearman")
    for split_name, results in split_results.items():
        for scorer_name in scorer_names:
            if scorer_name not in results:
                continue
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
    source_predictions_path: Path,
    policy_output_path: Path,
    predictions_output_path: Path,
    model_score_column: str,
    baseline_score_column: str,
    hybrid_frame,
    split_results: dict[str, dict[str, ScoreResult]],
    policy: dict[str, dict[str, object]],
    tops_min_subcategory_rows: int,
) -> dict[str, object]:
    split_row_counts = {
        split_name: int((hybrid_frame["split"] == split_name).sum())
        for split_name in ("train", "validation", "test")
    }
    split_query_group_counts = {
        split_name: {
            str(group): int(count)
            for group, count in hybrid_frame.loc[
                hybrid_frame["split"] == split_name,
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
            for group, count in hybrid_frame.loc[
                hybrid_frame["split"] == split_name,
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
            for subcategory, count in hybrid_frame.loc[
                hybrid_frame["split"] == split_name,
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
        blended_metrics = results[baseline_score_column].overall_metrics
        challenger_metrics = results[model_score_column].overall_metrics
        best_baseline_metrics = results[best_baseline_name].overall_metrics
        serialized_splits[split_name] = {
            "row_count": split_row_counts[split_name],
            "unique_asins": int(
                hybrid_frame.loc[hybrid_frame["split"] == split_name, "asin"].nunique()
            ),
            "query_group_counts": split_query_group_counts[split_name],
            "ranking_group_counts": split_ranking_group_counts[split_name],
            "subcategory_counts": split_subcategory_counts[split_name],
            "hybrid": {
                "overall_metrics": model_metrics,
                "per_query_group_metrics": results[PREDICTION_COLUMN].per_query_group_metrics,
                "tops_diagnostics": results[PREDICTION_COLUMN].tops_diagnostics,
            },
            "comparison": {
                "primary_metric": "ndcg@10",
                "best_baseline_name": best_baseline_name,
                "best_baseline_overall_metrics": best_baseline_metrics,
                "delta_vs_blended": build_metric_deltas(model_metrics, blended_metrics),
                "delta_vs_model": build_metric_deltas(model_metrics, challenger_metrics),
                "delta_vs_best_baseline": build_metric_deltas(
                    model_metrics,
                    best_baseline_metrics,
                ),
            },
        }

    return {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "source_predictions_path": str(source_predictions_path),
        "policy_output_path": str(policy_output_path),
        "predictions_output_path": str(predictions_output_path),
        "query_group_column": GROUP_COLUMN,
        "ranking_group_column": RANKING_GROUP_COLUMN,
        "target_column": TARGET_COLUMN,
        "model_score_column": model_score_column,
        "baseline_score_column": baseline_score_column,
        "hybrid_score_column": PREDICTION_COLUMN,
        "weight_grid": [float(value) for value in WEIGHT_GRID],
        "tops_min_subcategory_rows": int(tops_min_subcategory_rows),
        "policy": to_builtin(policy),
        "dataset_summary": {
            "total_rows": int(len(hybrid_frame)),
            "unique_asins": int(hybrid_frame["asin"].nunique()),
            "query_groups": sorted(hybrid_frame[GROUP_COLUMN].dropna().astype(str).unique().tolist()),
            "ranking_groups": sorted(
                hybrid_frame[RANKING_GROUP_COLUMN].dropna().astype(str).unique().tolist()
            ),
            "splits": split_row_counts,
        },
        "summary_rows": build_summary_rows(split_results),
        "splits": serialized_splits,
    }


def print_split_summary(
    split_name: str,
    results: dict[str, ScoreResult],
    model_score_column: str,
    baseline_score_column: str,
) -> None:
    hybrid_result = results[PREDICTION_COLUMN]
    best_baseline_name = choose_best_baseline(results)
    best_baseline_result = results[best_baseline_name]
    delta_vs_blended = build_metric_deltas(
        hybrid_result.overall_metrics,
        results[baseline_score_column].overall_metrics,
    )
    delta_vs_model = build_metric_deltas(
        hybrid_result.overall_metrics,
        results[model_score_column].overall_metrics,
    )

    print()
    print(f"{split_name.title()} split")
    print(
        "  hybrid_ranker: "
        f"rmse={hybrid_result.overall_metrics['rmse']:.4f}, "
        f"ndcg@10={hybrid_result.overall_metrics['ndcg@10']:.4f}, "
        f"ndcg@20={hybrid_result.overall_metrics['ndcg@20']:.4f}, "
        f"spearman={hybrid_result.overall_metrics['spearman']:.4f}"
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
    print(
        "  delta_vs_source_model: "
        f"rmse={delta_vs_model['rmse']:.4f}, "
        f"ndcg@10={delta_vs_model['ndcg@10']:.4f}, "
        f"ndcg@20={delta_vs_model['ndcg@20']:.4f}, "
        f"spearman={delta_vs_model['spearman']:.4f}"
    )
    print("  tops_fine_label_counts:")
    for subcategory in CORE_TOPS_SUBCATEGORIES:
        count = int(hybrid_result.tops_diagnostics["fine_subcategory_counts"].get(subcategory, 0))
        print(f"    {subcategory}: {count}")


def main() -> int:
    args = parse_args()
    if args.tops_min_subcategory_rows < 1:
        print("--tops-min-subcategory-rows must be at least 1.", file=sys.stderr)
        return 1

    source_predictions_path = normalize_path(args.source_predictions)
    policy_output_path = normalize_path(args.policy_output)
    predictions_output_path = normalize_path(args.predictions_output)
    results_output_path = normalize_path(args.results_output)

    if not source_predictions_path.exists():
        print(f"Source predictions file does not exist: {source_predictions_path}", file=sys.stderr)
        return 1

    try:
        import pandas as pd
    except ImportError as exc:
        print(
            "pandas is required to build the query-group hybrid ranker.\n"
            f"Import error: {exc}",
            file=sys.stderr,
        )
        return 1

    print("Loading tuned ranker predictions...")
    frame = pd.read_parquet(source_predictions_path)
    validate_required_columns(
        frame,
        REQUIRED_COLUMNS | {args.model_score_column, args.baseline_score_column},
    )
    if frame.empty:
        raise ValueError("Source predictions frame is empty.")
    if frame.duplicated(subset=["split", "asin"]).any():
        raise ValueError("Source predictions frame contains duplicate (split, asin) rows.")

    print("Selecting query-group hybrid weights on validation...")
    validation_frame = frame.loc[frame["split"] == "validation"].copy()
    if validation_frame.empty:
        raise ValueError("Source predictions frame must contain a validation split.")
    validation_ranked = add_rank_score_columns(
        validation_frame,
        (args.model_score_column, args.baseline_score_column),
    )
    policy = choose_weight_policy(
        validation_ranked,
        model_score_column=args.model_score_column,
        baseline_score_column=args.baseline_score_column,
    )

    print("Applying hybrid policy to all splits...")
    hybrid_frame = apply_weight_policy(
        frame,
        policy=policy,
        model_score_column=args.model_score_column,
        baseline_score_column=args.baseline_score_column,
    )
    if hybrid_frame[PREDICTION_COLUMN].isna().any():
        raise ValueError("Hybrid predictions contain missing values.")

    split_results: dict[str, dict[str, ScoreResult]] = {}
    for split_name in ("train", "validation", "test"):
        split_frame = hybrid_frame.loc[hybrid_frame["split"] == split_name].copy()
        split_results[split_name] = {
            PREDICTION_COLUMN: evaluate_score_column(
                split_frame,
                score_col=PREDICTION_COLUMN,
                min_subcategory_rows=args.tops_min_subcategory_rows,
            ),
            args.model_score_column: evaluate_score_column(
                split_frame,
                score_col=args.model_score_column,
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
        TARGET_COLUMN,
        args.model_score_column,
        args.baseline_score_column,
        "selected_baseline_weight",
        "selected_model_weight",
        PREDICTION_COLUMN,
        *BASELINE_COLUMNS,
    ]
    predictions_columns = list(dict.fromkeys(predictions_columns))
    predictions_frame = hybrid_frame.loc[:, predictions_columns].copy()
    split_order = {"train": 0, "validation": 1, "test": 2}
    predictions_frame["split_order"] = predictions_frame["split"].map(split_order)
    predictions_frame = predictions_frame.sort_values(
        ["split_order", GROUP_COLUMN, RANKING_GROUP_COLUMN, "subcategory", "asin"],
        kind="stable",
    ).drop(columns=["split_order"])
    predictions_frame = predictions_frame.reset_index(drop=True)

    print("Writing hybrid policy and artifacts...")
    policy_output_path.parent.mkdir(parents=True, exist_ok=True)
    policy_output_path.write_text(json.dumps(policy, indent=2, sort_keys=True), encoding="utf-8")

    predictions_output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_frame.to_parquet(predictions_output_path, index=False)

    results_output_path.parent.mkdir(parents=True, exist_ok=True)
    results_payload = serialize_results(
        source_predictions_path=source_predictions_path,
        policy_output_path=policy_output_path,
        predictions_output_path=predictions_output_path,
        model_score_column=args.model_score_column,
        baseline_score_column=args.baseline_score_column,
        hybrid_frame=predictions_frame,
        split_results=split_results,
        policy=policy,
        tops_min_subcategory_rows=args.tops_min_subcategory_rows,
    )
    results_output_path.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print()
    print("Query-group hybrid ranker build complete")
    print(f"Source predictions path: {source_predictions_path}")
    print(f"Policy path: {policy_output_path}")
    print(f"Predictions path: {predictions_output_path}")
    print(f"Results path: {results_output_path}")
    print_split_summary(
        "validation",
        split_results["validation"],
        model_score_column=args.model_score_column,
        baseline_score_column=args.baseline_score_column,
    )
    print_split_summary(
        "test",
        split_results["test"],
        model_score_column=args.model_score_column,
        baseline_score_column=args.baseline_score_column,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
