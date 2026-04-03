from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
MODELS_DIR = PROJECT_ROOT / "models"
for path in (EVALUATION_DIR, MODELS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ranking_metrics
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


DEFAULT_SOURCE_PREDICTIONS_PATH = Path("models/artifacts/query_group_hybrid_predictions.parquet")
DEFAULT_POLICY_OUTPUT_PATH = Path("models/artifacts/top10_champion_policy.json")
DEFAULT_PREDICTIONS_OUTPUT_PATH = Path("models/artifacts/top10_champion_predictions.parquet")
DEFAULT_RESULTS_OUTPUT_PATH = Path("evaluation/results/top10_champion_results.json")
DEFAULT_CANDIDATE_SCORE_COLUMNS = (
    "blended_score",
    "prediction_catboost_ranker_tuned",
    "prediction_query_group_hybrid",
)
DEFAULT_BASELINE_SCORE_COLUMN = "blended_score"
DEFAULT_NDCG10_SWITCH_THRESHOLD = 0.005
PREDICTION_COLUMN = "prediction_top10_champion"
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
            "Choose a conservative query-group champion focused on validation "
            "NDCG@10 using existing FitIQ scorer artifacts."
        )
    )
    parser.add_argument(
        "--source-predictions",
        type=Path,
        default=DEFAULT_SOURCE_PREDICTIONS_PATH,
        help="Path to a predictions parquet containing candidate scorer columns.",
    )
    parser.add_argument(
        "--policy-output",
        type=Path,
        default=DEFAULT_POLICY_OUTPUT_PATH,
        help="Destination JSON path for the learned query-group champion policy.",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=DEFAULT_PREDICTIONS_OUTPUT_PATH,
        help="Destination parquet path for champion predictions.",
    )
    parser.add_argument(
        "--results-output",
        type=Path,
        default=DEFAULT_RESULTS_OUTPUT_PATH,
        help="Destination JSON path for champion evaluation results.",
    )
    parser.add_argument(
        "--candidate-score-columns",
        nargs="+",
        default=list(DEFAULT_CANDIDATE_SCORE_COLUMNS),
        help="Candidate score columns to consider for each query group.",
    )
    parser.add_argument(
        "--baseline-score-column",
        type=str,
        default=DEFAULT_BASELINE_SCORE_COLUMN,
        help="Heuristic baseline score column used as the default fallback.",
    )
    parser.add_argument(
        "--ndcg10-switch-threshold",
        type=float,
        default=DEFAULT_NDCG10_SWITCH_THRESHOLD,
        help=(
            "Minimum validation NDCG@10 gain required before switching away from "
            "the baseline scorer within a query group."
        ),
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


def score_group(frame, score_column: str) -> dict[str, float]:
    truth = frame[TARGET_COLUMN].astype(float).tolist()
    scores = frame[score_column].astype(float).tolist()
    return {
        "ndcg@10": float(ranking_metrics.ndcg_at_k(truth, scores, k=10)),
        "ndcg@20": float(ranking_metrics.ndcg_at_k(truth, scores, k=20)),
        "spearman": float(ranking_metrics.spearman_rank_correlation(truth, scores)),
    }


def choose_group_policy(
    validation_frame,
    candidate_score_columns: tuple[str, ...],
    baseline_score_column: str,
    ndcg10_switch_threshold: float,
) -> dict[str, dict[str, object]]:
    policy: dict[str, dict[str, object]] = {}

    for query_group, group_frame in validation_frame.groupby(GROUP_COLUMN, sort=True):
        candidate_metrics = {
            score_column: score_group(group_frame, score_column)
            for score_column in candidate_score_columns
        }
        best_candidate = max(
            candidate_score_columns,
            key=lambda score_column: (
                candidate_metrics[score_column]["ndcg@10"],
                candidate_metrics[score_column]["ndcg@20"],
                candidate_metrics[score_column]["spearman"],
                1.0 if score_column == baseline_score_column else 0.0,
            ),
        )
        baseline_metrics = candidate_metrics[baseline_score_column]
        best_candidate_metrics = candidate_metrics[best_candidate]
        ndcg10_gain_vs_blended = float(
            best_candidate_metrics["ndcg@10"] - baseline_metrics["ndcg@10"]
        )

        selected_score_column = best_candidate
        selection_reason = "best_validation_ndcg10"
        if best_candidate == baseline_score_column:
            selection_reason = "baseline_already_best"
        elif ndcg10_gain_vs_blended < ndcg10_switch_threshold:
            selected_score_column = baseline_score_column
            selection_reason = "fallback_below_switch_threshold"

        policy[str(query_group)] = {
            "selected_score_column": str(selected_score_column),
            "selection_reason": selection_reason,
            "row_count": int(len(group_frame)),
            "baseline_score_column": str(baseline_score_column),
            "best_candidate_score_column": str(best_candidate),
            "ndcg10_switch_threshold": float(ndcg10_switch_threshold),
            "best_candidate_ndcg10_gain_vs_blended": ndcg10_gain_vs_blended,
            "validation_candidate_metrics": candidate_metrics,
            "selected_validation_metrics": candidate_metrics[selected_score_column],
        }

    return policy


def apply_group_policy(frame, policy: dict[str, dict[str, object]], baseline_score_column: str):
    scored = frame.copy()
    selected_columns = [
        str(policy.get(str(query_group), {}).get("selected_score_column", baseline_score_column))
        for query_group in scored[GROUP_COLUMN]
    ]
    scored["selected_score_column"] = selected_columns
    scored[PREDICTION_COLUMN] = [
        float(getattr(row, score_column))
        for row, score_column in zip(scored.itertuples(index=False), selected_columns, strict=True)
    ]
    return scored


def build_summary_rows(split_results: dict[str, dict[str, ScoreResult]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    metric_names = ("rmse", "ndcg@10", "ndcg@20", "spearman")
    scorer_names: list[str] = []
    for split_name in ("train", "validation", "test"):
        if split_name in split_results:
            scorer_names = list(split_results[split_name].keys())
            break
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
    candidate_score_columns: tuple[str, ...],
    baseline_score_column: str,
    ndcg10_switch_threshold: float,
    tops_min_subcategory_rows: int,
    scored_frame,
    split_results: dict[str, dict[str, ScoreResult]],
    policy: dict[str, dict[str, object]],
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
    policy_usage_counts = {
        split_name: {
            str(score_column): int(count)
            for score_column, count in scored_frame.loc[
                scored_frame["split"] == split_name,
                "selected_score_column",
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
        champion_metrics = results[PREDICTION_COLUMN].overall_metrics
        blended_metrics = results[baseline_score_column].overall_metrics
        best_baseline_metrics = results[best_baseline_name].overall_metrics
        serialized_splits[split_name] = {
            "row_count": split_row_counts[split_name],
            "unique_asins": int(
                scored_frame.loc[scored_frame["split"] == split_name, "asin"].nunique()
            ),
            "query_group_counts": split_query_group_counts[split_name],
            "ranking_group_counts": split_ranking_group_counts[split_name],
            "subcategory_counts": split_subcategory_counts[split_name],
            "selected_score_column_counts": policy_usage_counts[split_name],
            "model": {
                "overall_metrics": champion_metrics,
                "per_query_group_metrics": results[PREDICTION_COLUMN].per_query_group_metrics,
                "tops_diagnostics": results[PREDICTION_COLUMN].tops_diagnostics,
            },
            "candidates": {
                score_column: {
                    "overall_metrics": results[score_column].overall_metrics,
                    "per_query_group_metrics": results[score_column].per_query_group_metrics,
                    "tops_diagnostics": results[score_column].tops_diagnostics,
                }
                for score_column in candidate_score_columns
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
                "delta_vs_blended": build_metric_deltas(champion_metrics, blended_metrics),
                "delta_vs_best_baseline": build_metric_deltas(
                    champion_metrics,
                    best_baseline_metrics,
                ),
                "deltas_vs_candidates": {
                    score_column: build_metric_deltas(
                        champion_metrics,
                        results[score_column].overall_metrics,
                    )
                    for score_column in candidate_score_columns
                },
            },
        }

    return {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "source_predictions_path": str(source_predictions_path),
        "policy_path": str(policy_output_path),
        "predictions_path": str(predictions_output_path),
        "model_type": "ValidationGatedTop10Champion",
        "target_column": TARGET_COLUMN,
        "query_group_column": GROUP_COLUMN,
        "ranking_group_column": RANKING_GROUP_COLUMN,
        "baseline_score_column": baseline_score_column,
        "candidate_score_columns": list(candidate_score_columns),
        "ndcg10_switch_threshold": float(ndcg10_switch_threshold),
        "tops_min_subcategory_rows": int(tops_min_subcategory_rows),
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
        "selected_policy": to_builtin(policy),
        "summary_rows": build_summary_rows(split_results),
        "splits": serialized_splits,
    }


def print_split_summary(split_name: str, results: dict[str, ScoreResult]) -> None:
    champion_result = results[PREDICTION_COLUMN]
    best_baseline_name = choose_best_baseline(results)
    best_baseline_result = results[best_baseline_name]
    delta_vs_blended = build_metric_deltas(
        champion_result.overall_metrics,
        results["blended_score"].overall_metrics,
    )

    print()
    print(f"{split_name.title()} split")
    print(
        "  top10_champion: "
        f"rmse={champion_result.overall_metrics['rmse']:.4f}, "
        f"ndcg@10={champion_result.overall_metrics['ndcg@10']:.4f}, "
        f"ndcg@20={champion_result.overall_metrics['ndcg@20']:.4f}, "
        f"spearman={champion_result.overall_metrics['spearman']:.4f}"
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
    print("  tops_fine_label_counts:")
    for subcategory in CORE_TOPS_SUBCATEGORIES:
        count = int(champion_result.tops_diagnostics["fine_subcategory_counts"].get(subcategory, 0))
        print(f"    {subcategory}: {count}")


def main() -> int:
    args = parse_args()
    if args.ndcg10_switch_threshold < 0.0:
        print("--ndcg10-switch-threshold must be non-negative.", file=sys.stderr)
        return 1
    if args.tops_min_subcategory_rows < 1:
        print("--tops-min-subcategory-rows must be at least 1.", file=sys.stderr)
        return 1

    source_predictions_path = normalize_path(args.source_predictions)
    policy_output_path = normalize_path(args.policy_output)
    predictions_output_path = normalize_path(args.predictions_output)
    results_output_path = normalize_path(args.results_output)
    candidate_score_columns = tuple(dict.fromkeys(args.candidate_score_columns))
    if args.baseline_score_column not in candidate_score_columns:
        print(
            "--baseline-score-column must also appear in --candidate-score-columns.",
            file=sys.stderr,
        )
        return 1
    if not source_predictions_path.exists():
        print(f"Predictions file does not exist: {source_predictions_path}", file=sys.stderr)
        return 1

    try:
        import pandas as pd
    except ImportError as exc:
        print(
            "pandas is required to build the top-10 champion policy.\n"
            f"Import error: {exc}",
            file=sys.stderr,
        )
        return 1

    print("Loading candidate predictions...")
    frame = pd.read_parquet(source_predictions_path)
    validate_required_columns(
        frame,
        REQUIRED_COLUMNS | set(candidate_score_columns) | set(BASELINE_COLUMNS),
    )
    if frame[["split", "asin"]].duplicated().any():
        raise ValueError("Predictions frame contains duplicate (split, asin) rows.")

    validation_frame = frame.loc[frame["split"] == "validation"].copy()
    if validation_frame.empty:
        raise ValueError("Validation split is empty; cannot learn champion policy.")

    print("Selecting validation-gated query-group policy...")
    policy = choose_group_policy(
        validation_frame=validation_frame,
        candidate_score_columns=candidate_score_columns,
        baseline_score_column=args.baseline_score_column,
        ndcg10_switch_threshold=float(args.ndcg10_switch_threshold),
    )

    print("Applying policy to all splits...")
    scored_frame = apply_group_policy(
        frame=frame,
        policy=policy,
        baseline_score_column=args.baseline_score_column,
    )

    score_columns = [PREDICTION_COLUMN, *candidate_score_columns]
    for baseline_column in BASELINE_COLUMNS:
        if baseline_column not in score_columns:
            score_columns.append(baseline_column)

    split_results: dict[str, dict[str, ScoreResult]] = {}
    for split_name in ("train", "validation", "test"):
        split_frame = scored_frame.loc[scored_frame["split"] == split_name].copy()
        split_results[split_name] = {
            score_column: evaluate_score_column(
                split_frame,
                score_col=score_column,
                min_subcategory_rows=args.tops_min_subcategory_rows,
            )
            for score_column in score_columns
        }

    print("Writing champion artifacts...")
    policy_output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_output_path.parent.mkdir(parents=True, exist_ok=True)
    results_output_path.parent.mkdir(parents=True, exist_ok=True)
    policy_output_path.write_text(json.dumps(to_builtin(policy), indent=2), encoding="utf-8")
    scored_frame.to_parquet(predictions_output_path, index=False)

    results_payload = serialize_results(
        source_predictions_path=source_predictions_path,
        policy_output_path=policy_output_path,
        predictions_output_path=predictions_output_path,
        candidate_score_columns=candidate_score_columns,
        baseline_score_column=args.baseline_score_column,
        ndcg10_switch_threshold=float(args.ndcg10_switch_threshold),
        tops_min_subcategory_rows=args.tops_min_subcategory_rows,
        scored_frame=scored_frame,
        split_results=split_results,
        policy=policy,
    )
    results_output_path.write_text(
        json.dumps(to_builtin(results_payload), indent=2),
        encoding="utf-8",
    )

    print("Top-10 champion policy complete")
    print(f"  policy_path: {policy_output_path}")
    print(f"  predictions_path: {predictions_output_path}")
    print(f"  results_path: {results_output_path}")
    for split_name in ("validation", "test"):
        print_split_summary(split_name, split_results[split_name])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
