from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

import ranking_metrics


DEFAULT_PREDICTIONS_PATH = Path("models/artifacts/stacked_catboost_ranker_predictions.parquet")
DEFAULT_OUTPUT_PATH = Path("evaluation/results/bootstrap_model_comparison.json")
DEFAULT_MODEL_SCORE_COLUMN = "prediction_catboost_ranker_stacked"
DEFAULT_BASELINE_SCORE_COLUMN = "blended_score"
DEFAULT_RANDOM_SEED = 20260401
DEFAULT_BOOTSTRAP_ITERATIONS = 5000
DEFAULT_SPLITS = ("validation", "test")
QUERY_GROUP_COLUMN = "query_group_v2"
TARGET_COLUMN = "target_score"
REQUIRED_COLUMNS = {
    "split",
    QUERY_GROUP_COLUMN,
    TARGET_COLUMN,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap model-versus-baseline ranking metric deltas over query groups."
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
        help="Destination JSON path for bootstrap comparison results.",
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
        "--bootstrap-iterations",
        type=int,
        default=DEFAULT_BOOTSTRAP_ITERATIONS,
        help="Number of bootstrap resamples over query groups.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for bootstrap resampling.",
    )
    return parser.parse_args()


def normalize_path(path: Path) -> Path:
    return path.resolve()


def validate_required_columns(frame, required_columns: set[str]) -> None:
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"Predictions frame is missing required columns: {', '.join(missing)}")


def compute_group_metric_vectors(
    frame,
    model_score_column: str,
    baseline_score_column: str,
) -> dict[str, dict[str, list[float]]]:
    vectors = {
        "ndcg@10": {"model": [], "baseline": [], "delta": []},
        "ndcg@20": {"model": [], "baseline": [], "delta": []},
        "spearman": {"model": [], "baseline": [], "delta": []},
    }
    for _, group_frame in frame.groupby(QUERY_GROUP_COLUMN, sort=True):
        truth = group_frame[TARGET_COLUMN].astype(float).tolist()
        model_scores = group_frame[model_score_column].astype(float).tolist()
        baseline_scores = group_frame[baseline_score_column].astype(float).tolist()
        metric_values = {
            "ndcg@10": (
                ranking_metrics.ndcg_at_k(truth, model_scores, k=10),
                ranking_metrics.ndcg_at_k(truth, baseline_scores, k=10),
            ),
            "ndcg@20": (
                ranking_metrics.ndcg_at_k(truth, model_scores, k=20),
                ranking_metrics.ndcg_at_k(truth, baseline_scores, k=20),
            ),
            "spearman": (
                ranking_metrics.spearman_rank_correlation(truth, model_scores),
                ranking_metrics.spearman_rank_correlation(truth, baseline_scores),
            ),
        }
        for metric_name, (model_value, baseline_value) in metric_values.items():
            vectors[metric_name]["model"].append(float(model_value))
            vectors[metric_name]["baseline"].append(float(baseline_value))
            vectors[metric_name]["delta"].append(float(model_value - baseline_value))
    return vectors


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    index = int(round((len(sorted_values) - 1) * q))
    index = max(0, min(index, len(sorted_values) - 1))
    return float(sorted_values[index])


def bootstrap_delta_summary(
    deltas: list[float],
    iterations: int,
    rng: random.Random,
) -> dict[str, float]:
    if not deltas:
        return {
            "observed_mean_delta": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "p_model_better": 0.0,
        }

    if len(deltas) == 1:
        observed = float(deltas[0])
        probability = 1.0 if observed > 0.0 else 0.0
        return {
            "observed_mean_delta": observed,
            "ci_lower": observed,
            "ci_upper": observed,
            "p_model_better": probability,
        }

    bootstrap_means: list[float] = []
    for _ in range(iterations):
        sample = [deltas[rng.randrange(len(deltas))] for _ in range(len(deltas))]
        bootstrap_means.append(float(statistics.fmean(sample)))
    bootstrap_means.sort()
    observed_mean = float(statistics.fmean(deltas))
    p_model_better = float(
        sum(value > 0.0 for value in bootstrap_means) / len(bootstrap_means)
    )
    return {
        "observed_mean_delta": observed_mean,
        "ci_lower": percentile(bootstrap_means, 0.025),
        "ci_upper": percentile(bootstrap_means, 0.975),
        "p_model_better": p_model_better,
    }


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


def print_split_summary(split_name: str, split_results: dict[str, object]) -> None:
    print()
    print(f"{split_name.title()} split")
    print(f"  query_groups: {split_results['query_group_count']}")
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
    print("  bootstrap_delta_summary:")
    for metric_name, payload in split_results["bootstrap_metrics"].items():
        print(
            f"    {metric_name}: "
            f"mean_delta={payload['observed_mean_delta']:.4f}, "
            f"ci=[{payload['ci_lower']:.4f}, {payload['ci_upper']:.4f}], "
            f"p_model_better={payload['p_model_better']:.3f}"
        )


def main() -> int:
    args = parse_args()
    if args.bootstrap_iterations < 1:
        print("--bootstrap-iterations must be at least 1.", file=sys.stderr)
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
            "pandas is required to run bootstrap comparison.\n"
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
        "target_column": TARGET_COLUMN,
        "bootstrap_iterations": int(args.bootstrap_iterations),
        "random_seed": int(args.random_seed),
        "splits": {},
    }

    for split_name in DEFAULT_SPLITS:
        split_frame = frame.loc[frame["split"] == split_name].copy()
        if split_frame.empty:
            raise ValueError(f"Predictions artifact does not contain split: {split_name}")

        vectors = compute_group_metric_vectors(
            split_frame,
            model_score_column=args.model_score_column,
            baseline_score_column=args.baseline_score_column,
        )
        rng = random.Random(args.random_seed + len(split_frame) + len(split_name))
        bootstrap_metrics = {
            metric_name: bootstrap_delta_summary(
                payload["delta"],
                iterations=args.bootstrap_iterations,
                rng=rng,
            )
            for metric_name, payload in vectors.items()
        }
        results["splits"][split_name] = {
            "row_count": int(len(split_frame)),
            "unique_asins": int(split_frame["asin"].nunique()),
            "query_group_count": int(split_frame[QUERY_GROUP_COLUMN].nunique()),
            "model_overall_metrics": compute_overall_metrics(
                split_frame, score_column=args.model_score_column
            ),
            "baseline_overall_metrics": compute_overall_metrics(
                split_frame, score_column=args.baseline_score_column
            ),
            "bootstrap_metrics": bootstrap_metrics,
            "query_group_counts": {
                str(group): int(count)
                for group, count in split_frame[QUERY_GROUP_COLUMN]
                .value_counts()
                .sort_index()
                .items()
            },
        }

    print("Writing bootstrap comparison results...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    print()
    print("Bootstrap comparison complete")
    print(f"Predictions path: {predictions_path}")
    print(f"Output path: {output_path}")
    for split_name in DEFAULT_SPLITS:
        print_split_summary(split_name, results["splits"][split_name])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
