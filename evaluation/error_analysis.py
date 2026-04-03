from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import ranking_metrics


DEFAULT_PREDICTIONS_PATH = Path("models/artifacts/stacked_catboost_ranker_predictions.parquet")
DEFAULT_OUTPUT_PATH = Path("evaluation/results/stacked_ranker_error_analysis.json")
DEFAULT_MODEL_SCORE_COLUMN = "prediction_catboost_ranker_stacked"
DEFAULT_BASELINE_SCORE_COLUMN = "blended_score"
DEFAULT_TOP_N = 5
QUERY_GROUP_COLUMN = "query_group_v2"
RANKING_GROUP_COLUMN = "ranking_group"
TARGET_COLUMN = "target_score"
REQUIRED_COLUMNS = {
    "split",
    "asin",
    QUERY_GROUP_COLUMN,
    RANKING_GROUP_COLUMN,
    "subcategory",
    TARGET_COLUMN,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze where the stacked ranker wins and loses versus blended_score."
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=DEFAULT_PREDICTIONS_PATH,
        help="Path to a predictions parquet containing model and baseline scores.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination JSON path for the error analysis report.",
    )
    parser.add_argument(
        "--model-score-column",
        type=str,
        default=DEFAULT_MODEL_SCORE_COLUMN,
        help="Column name for the model score in the predictions parquet.",
    )
    parser.add_argument(
        "--baseline-score-column",
        type=str,
        default=DEFAULT_BASELINE_SCORE_COLUMN,
        help="Column name for the baseline score in the predictions parquet.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help="Number of query-group and row-level examples to report in each direction.",
    )
    return parser.parse_args()


def normalize_path(path: Path) -> Path:
    return path.resolve()


def validate_required_columns(frame, required_columns: set[str]) -> None:
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"Predictions frame is missing required columns: {', '.join(missing)}")


def compute_overall_metrics(frame, score_column: str) -> dict[str, float]:
    return {
        "ndcg@10": float(
            ranking_metrics.grouped_ndcg_at_k(
                frame,
                group_col=QUERY_GROUP_COLUMN,
                truth_col=TARGET_COLUMN,
                score_col=score_column,
                k=10,
            )
        ),
        "ndcg@20": float(
            ranking_metrics.grouped_ndcg_at_k(
                frame,
                group_col=QUERY_GROUP_COLUMN,
                truth_col=TARGET_COLUMN,
                score_col=score_column,
                k=20,
            )
        ),
        "spearman": float(
            ranking_metrics.grouped_spearman(
                frame,
                group_col=QUERY_GROUP_COLUMN,
                truth_col=TARGET_COLUMN,
                score_col=score_column,
            )
        ),
    }


def compute_query_group_rows(frame, model_score_column: str, baseline_score_column: str):
    rows: list[dict[str, object]] = []
    for query_group, group_frame in frame.groupby(QUERY_GROUP_COLUMN, sort=True):
        truth = group_frame[TARGET_COLUMN].astype(float).tolist()
        model_scores = group_frame[model_score_column].astype(float).tolist()
        baseline_scores = group_frame[baseline_score_column].astype(float).tolist()
        model_metrics = {
            "ndcg@10": float(ranking_metrics.ndcg_at_k(truth, model_scores, k=10)),
            "ndcg@20": float(ranking_metrics.ndcg_at_k(truth, model_scores, k=20)),
            "spearman": float(
                ranking_metrics.spearman_rank_correlation(truth, model_scores)
            ),
        }
        baseline_metrics = {
            "ndcg@10": float(ranking_metrics.ndcg_at_k(truth, baseline_scores, k=10)),
            "ndcg@20": float(ranking_metrics.ndcg_at_k(truth, baseline_scores, k=20)),
            "spearman": float(
                ranking_metrics.spearman_rank_correlation(truth, baseline_scores)
            ),
        }
        rows.append(
            {
                "query_group": str(query_group),
                "ranking_group": str(group_frame[RANKING_GROUP_COLUMN].iloc[0]),
                "row_count": int(len(group_frame)),
                "model_metrics": model_metrics,
                "baseline_metrics": baseline_metrics,
                "delta": {
                    metric_name: float(model_metrics[metric_name] - baseline_metrics[metric_name])
                    for metric_name in ("ndcg@10", "ndcg@20", "spearman")
                },
            }
        )
    return rows


def compute_rank_disagreement_rows(
    frame,
    model_score_column: str,
    baseline_score_column: str,
    top_n: int,
) -> dict[str, list[dict[str, object]]]:
    enriched = frame.copy()
    enriched["target_rank"] = enriched.groupby(QUERY_GROUP_COLUMN)[TARGET_COLUMN].rank(
        method="first",
        ascending=False,
    )
    enriched["model_rank"] = enriched.groupby(QUERY_GROUP_COLUMN)[model_score_column].rank(
        method="first",
        ascending=False,
    )
    enriched["baseline_rank"] = enriched.groupby(QUERY_GROUP_COLUMN)[baseline_score_column].rank(
        method="first",
        ascending=False,
    )
    enriched["model_abs_rank_error"] = (enriched["model_rank"] - enriched["target_rank"]).abs()
    enriched["baseline_abs_rank_error"] = (
        enriched["baseline_rank"] - enriched["target_rank"]
    ).abs()
    enriched["rank_error_delta"] = (
        enriched["baseline_abs_rank_error"] - enriched["model_abs_rank_error"]
    )

    select_columns = [
        "asin",
        QUERY_GROUP_COLUMN,
        RANKING_GROUP_COLUMN,
        "subcategory",
        TARGET_COLUMN,
        model_score_column,
        baseline_score_column,
        "target_rank",
        "model_rank",
        "baseline_rank",
        "rank_error_delta",
    ]
    biggest_model_wins = (
        enriched.sort_values(["rank_error_delta", TARGET_COLUMN], ascending=[False, False])
        .loc[:, select_columns]
        .head(top_n)
        .to_dict(orient="records")
    )
    biggest_model_losses = (
        enriched.sort_values(["rank_error_delta", TARGET_COLUMN], ascending=[True, False])
        .loc[:, select_columns]
        .head(top_n)
        .to_dict(orient="records")
    )
    return {
        "biggest_model_wins": biggest_model_wins,
        "biggest_model_losses": biggest_model_losses,
    }


def print_split_summary(split_name: str, split_results: dict[str, object]) -> None:
    print()
    print(f"{split_name.title()} split")
    print(
        "  model_overall: "
        f"ndcg@10={split_results['model_overall_metrics']['ndcg@10']:.4f}, "
        f"ndcg@20={split_results['model_overall_metrics']['ndcg@20']:.4f}, "
        f"spearman={split_results['model_overall_metrics']['spearman']:.4f}"
    )
    print(
        "  baseline_overall: "
        f"ndcg@10={split_results['baseline_overall_metrics']['ndcg@10']:.4f}, "
        f"ndcg@20={split_results['baseline_overall_metrics']['ndcg@20']:.4f}, "
        f"spearman={split_results['baseline_overall_metrics']['spearman']:.4f}"
    )

    print("  strongest_query_group_wins:")
    for row in split_results["top_query_group_wins"]:
        print(
            f"    {row['query_group']}: "
            f"delta_ndcg@10={row['delta']['ndcg@10']:.4f}, "
            f"delta_ndcg@20={row['delta']['ndcg@20']:.4f}, "
            f"delta_spearman={row['delta']['spearman']:.4f}"
        )

    print("  strongest_query_group_losses:")
    for row in split_results["top_query_group_losses"]:
        print(
            f"    {row['query_group']}: "
            f"delta_ndcg@10={row['delta']['ndcg@10']:.4f}, "
            f"delta_ndcg@20={row['delta']['ndcg@20']:.4f}, "
            f"delta_spearman={row['delta']['spearman']:.4f}"
        )


def main() -> int:
    args = parse_args()
    if args.top_n < 1:
        print("--top-n must be at least 1.", file=sys.stderr)
        return 1

    predictions_path = normalize_path(args.predictions_path)
    output_path = normalize_path(args.output_path)
    if not predictions_path.exists():
        print(f"Predictions file does not exist: {predictions_path}", file=sys.stderr)
        return 1

    try:
        import pandas as pd
    except ImportError as exc:
        print(
            "pandas is required to run error analysis.\n"
            f"Import error: {exc}",
            file=sys.stderr,
        )
        return 1

    print("Loading predictions artifact...")
    frame = pd.read_parquet(predictions_path)
    validate_required_columns(
        frame,
        REQUIRED_COLUMNS | {args.model_score_column, args.baseline_score_column},
    )

    results: dict[str, object] = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "predictions_path": str(predictions_path),
        "model_score_column": args.model_score_column,
        "baseline_score_column": args.baseline_score_column,
        "query_group_column": QUERY_GROUP_COLUMN,
        "ranking_group_column": RANKING_GROUP_COLUMN,
        "target_column": TARGET_COLUMN,
        "splits": {},
    }

    for split_name in ("validation", "test"):
        split_frame = frame.loc[frame["split"] == split_name].copy()
        if split_frame.empty:
            raise ValueError(f"Predictions artifact does not contain split: {split_name}")

        query_group_rows = compute_query_group_rows(
            split_frame,
            model_score_column=args.model_score_column,
            baseline_score_column=args.baseline_score_column,
        )
        sorted_wins = sorted(
            query_group_rows,
            key=lambda row: row["delta"]["ndcg@10"],
            reverse=True,
        )
        sorted_losses = sorted(
            query_group_rows,
            key=lambda row: row["delta"]["ndcg@10"],
        )
        disagreement_rows = compute_rank_disagreement_rows(
            split_frame,
            model_score_column=args.model_score_column,
            baseline_score_column=args.baseline_score_column,
            top_n=args.top_n,
        )

        results["splits"][split_name] = {
            "row_count": int(len(split_frame)),
            "unique_asins": int(split_frame["asin"].nunique()),
            "query_group_count": int(split_frame[QUERY_GROUP_COLUMN].nunique()),
            "model_overall_metrics": compute_overall_metrics(
                split_frame,
                score_column=args.model_score_column,
            ),
            "baseline_overall_metrics": compute_overall_metrics(
                split_frame,
                score_column=args.baseline_score_column,
            ),
            "query_group_rows": query_group_rows,
            "top_query_group_wins": sorted_wins[: args.top_n],
            "top_query_group_losses": sorted_losses[: args.top_n],
            **disagreement_rows,
        }

    print("Writing error analysis report...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    print()
    print("Error analysis complete")
    print(f"Predictions path: {predictions_path}")
    print(f"Output path: {output_path}")
    for split_name in ("validation", "test"):
        print_split_summary(split_name, results["splits"][split_name])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
