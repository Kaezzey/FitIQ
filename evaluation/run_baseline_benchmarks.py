from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import ranking_metrics


DEFAULT_DATASET_PATH = Path("data/processed/ranking_eval_dataset.parquet")
DEFAULT_OUTPUT_PATH = Path("evaluation/results/baseline_benchmark_results.json")
DEFAULT_RANDOM_SEED = 20260401
DEFAULT_TOPS_MIN_ROWS = 25
REQUIRED_COLUMNS = {
    "split",
    "asin",
    "ranking_group",
    "subcategory",
    "mean_rating",
    "fit_complaint_rate",
    "review_count",
    "target_score",
}
BASELINE_NAMES = (
    "random_score",
    "mean_rating_score",
    "fit_score",
    "review_volume_score",
    "blended_score",
)
CORE_TOPS_SUBCATEGORIES = ("t_shirt", "shirt_top", "sweater", "unknown")


@dataclass(frozen=True)
class BaselineResult:
    name: str
    overall_metrics: dict[str, float]
    per_ranking_group_metrics: dict[str, dict[str, float]]
    tops_diagnostics: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run heuristic baseline benchmarks on the ranking evaluation dataset."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to data/processed/ranking_eval_dataset.parquet.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination JSON path for baseline benchmark results.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Seed used for the reproducible random baseline.",
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
        scored["review_count"].astype(float) / (scored["review_count"].astype(float) + 5.0)
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


def compute_overall_metrics(frame, score_col: str) -> dict[str, float]:
    return {
        "ndcg@10": float(
            ranking_metrics.grouped_ndcg_at_k(
                frame,
                group_col="ranking_group",
                truth_col="target_score",
                score_col=score_col,
                k=10,
            )
        ),
        "ndcg@20": float(
            ranking_metrics.grouped_ndcg_at_k(
                frame,
                group_col="ranking_group",
                truth_col="target_score",
                score_col=score_col,
                k=20,
            )
        ),
        "spearman": float(
            ranking_metrics.grouped_spearman(
                frame,
                group_col="ranking_group",
                truth_col="target_score",
                score_col=score_col,
            )
        ),
    }


def compute_per_ranking_group_metrics(frame, score_col: str) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for ranking_group, group_frame in frame.groupby("ranking_group", sort=True):
        truth = group_frame["target_score"].astype(float).tolist()
        scores = group_frame[score_col].astype(float).tolist()
        metrics[str(ranking_group)] = {
            "row_count": int(len(group_frame)),
            "unique_asins": int(group_frame["asin"].nunique()),
            "ndcg@10": float(ranking_metrics.ndcg_at_k(truth, scores, k=10)),
            "ndcg@20": float(ranking_metrics.ndcg_at_k(truth, scores, k=20)),
            "spearman": float(ranking_metrics.spearman_rank_correlation(truth, scores)),
        }
    return metrics


def compute_tops_diagnostics(frame, score_col: str, min_subcategory_rows: int) -> dict[str, object]:
    tops = frame.loc[frame["ranking_group"] == "tops"].copy()
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
        truth = sub_frame["target_score"].astype(float).tolist()
        scores = sub_frame[score_col].astype(float).tolist()
        metric_slices[subcategory] = {
            "row_count": int(len(sub_frame)),
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


def evaluate_split(frame, min_subcategory_rows: int) -> dict[str, BaselineResult]:
    split_results: dict[str, BaselineResult] = {}
    for baseline_name in BASELINE_NAMES:
        split_results[baseline_name] = BaselineResult(
            name=baseline_name,
            overall_metrics=compute_overall_metrics(frame, score_col=baseline_name),
            per_ranking_group_metrics=compute_per_ranking_group_metrics(
                frame, score_col=baseline_name
            ),
            tops_diagnostics=compute_tops_diagnostics(
                frame,
                score_col=baseline_name,
                min_subcategory_rows=min_subcategory_rows,
            ),
        )
    return split_results


def build_summary_rows(results_by_split: dict[str, dict[str, BaselineResult]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    metric_names = ("ndcg@10", "ndcg@20", "spearman")
    for split_name, baseline_results in results_by_split.items():
        for baseline_name, result in baseline_results.items():
            for metric_name in metric_names:
                rows.append(
                    {
                        "split": split_name,
                        "baseline": baseline_name,
                        "metric": metric_name,
                        "overall_score": float(result.overall_metrics[metric_name]),
                    }
                )
    return rows


def serialize_results(
    dataset_path: Path,
    random_seed: int,
    tops_min_subcategory_rows: int,
    frame,
    results_by_split: dict[str, dict[str, BaselineResult]],
) -> dict[str, object]:
    split_row_counts = {
        split_name: int((frame["split"] == split_name).sum())
        for split_name in ("train", "validation", "test")
    }
    split_group_counts = {
        split_name: {
            str(group): int(count)
            for group, count in frame.loc[frame["split"] == split_name, "ranking_group"]
            .value_counts()
            .sort_index()
            .items()
        }
        for split_name in ("train", "validation", "test")
    }
    split_subcategory_counts = {
        split_name: {
            str(subcategory): int(count)
            for subcategory, count in frame.loc[frame["split"] == split_name, "subcategory"]
            .value_counts()
            .sort_index()
            .items()
        }
        for split_name in ("train", "validation", "test")
    }

    serialized_splits: dict[str, object] = {}
    for split_name, baseline_results in results_by_split.items():
        serialized_splits[split_name] = {
            "row_count": split_row_counts[split_name],
            "ranking_group_counts": split_group_counts[split_name],
            "subcategory_counts": split_subcategory_counts[split_name],
            "baselines": {
                baseline_name: {
                    "overall_metrics": result.overall_metrics,
                    "per_ranking_group_metrics": result.per_ranking_group_metrics,
                    "tops_diagnostics": result.tops_diagnostics,
                }
                for baseline_name, result in baseline_results.items()
            },
        }

    return {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "dataset_path": str(dataset_path),
        "random_seed": int(random_seed),
        "tops_min_subcategory_rows": int(tops_min_subcategory_rows),
        "dataset_summary": {
            "total_rows": int(len(frame)),
            "unique_asins": int(frame["asin"].nunique()),
            "ranking_groups": sorted(frame["ranking_group"].dropna().astype(str).unique().tolist()),
            "splits": split_row_counts,
        },
        "summary_rows": build_summary_rows(results_by_split),
        "splits": serialized_splits,
    }


def print_split_summary(split_name: str, result_bundle: dict[str, BaselineResult], frame) -> None:
    split_frame = frame.loc[frame["split"] == split_name]
    print()
    print(f"{split_name.title()} split")
    print(f"  rows: {len(split_frame)}")
    print(f"  unique_asins: {split_frame['asin'].nunique()}")
    ranking_group_counts = split_frame["ranking_group"].value_counts().sort_index()
    print("  ranking_group_counts:")
    for ranking_group, count in ranking_group_counts.items():
        print(f"    {ranking_group}: {int(count)}")

    for baseline_name in BASELINE_NAMES:
        result = result_bundle[baseline_name]
        print(
            f"  {baseline_name}: "
            f"ndcg@10={result.overall_metrics['ndcg@10']:.4f}, "
            f"ndcg@20={result.overall_metrics['ndcg@20']:.4f}, "
            f"spearman={result.overall_metrics['spearman']:.4f}"
        )

    tops_diagnostics = result_bundle["blended_score"].tops_diagnostics
    print("  tops_fine_label_counts:")
    for subcategory in CORE_TOPS_SUBCATEGORIES:
        print(
            f"    {subcategory}: "
            f"{int(tops_diagnostics['fine_subcategory_counts'].get(subcategory, 0))}"
        )


def main() -> int:
    args = parse_args()
    if args.tops_min_subcategory_rows < 1:
        print("--tops-min-subcategory-rows must be at least 1.", file=sys.stderr)
        return 1

    dataset_path = normalize_path(args.dataset_path)
    output_path = normalize_path(args.output_path)
    if not dataset_path.exists():
        print(f"Dataset file does not exist: {dataset_path}", file=sys.stderr)
        return 1

    try:
        import pandas as pd
    except ImportError as exc:
        print(
            "pandas is required to run baseline benchmarks. Install it in the "
            "project Python environment, then rerun the script.\n"
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

    print("Computing baseline scores...")
    frame = compute_baseline_scores(frame, seed=args.random_seed)

    for baseline_name in BASELINE_NAMES:
        if frame[baseline_name].isna().any():
            raise ValueError(f"Baseline {baseline_name} contains missing scores.")
        invalid = (frame[baseline_name] < 0.0) | (frame[baseline_name] > 1.0)
        if invalid.any():
            raise ValueError(f"Baseline {baseline_name} must stay within [0, 1].")

    print("Evaluating baselines by split...")
    results_by_split: dict[str, dict[str, BaselineResult]] = {}
    for split_name in ("train", "validation", "test"):
        split_frame = frame.loc[frame["split"] == split_name].copy()
        if split_frame.empty:
            raise ValueError(f"Split {split_name} is empty in the ranking evaluation dataset.")
        results_by_split[split_name] = evaluate_split(
            split_frame,
            min_subcategory_rows=args.tops_min_subcategory_rows,
        )

    validation_random = results_by_split["validation"]["random_score"].overall_metrics["ndcg@10"]
    if not any(
        results_by_split["validation"][baseline].overall_metrics["ndcg@10"] > validation_random
        for baseline in BASELINE_NAMES
        if baseline != "random_score"
    ):
        raise ValueError("No deterministic validation baseline outperformed random_score on NDCG@10.")

    serialized_results = serialize_results(
        dataset_path=dataset_path,
        random_seed=args.random_seed,
        tops_min_subcategory_rows=args.tops_min_subcategory_rows,
        frame=frame,
        results_by_split=results_by_split,
    )

    print("Writing baseline benchmark results...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(serialized_results, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print()
    print("Baseline benchmark summary")
    print(f"Dataset path: {dataset_path}")
    print(f"Output path: {output_path}")
    print(f"Random seed: {args.random_seed}")
    for split_name in ("train", "validation", "test"):
        print_split_summary(split_name, results_by_split[split_name], frame)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
