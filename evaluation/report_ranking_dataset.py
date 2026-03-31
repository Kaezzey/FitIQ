from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path


DEFAULT_DATASET_PATH = Path("data/processed/ranking_eval_dataset.parquet")
CORE_SUBCATEGORIES = ("t_shirt", "shirt_top", "pants", "jacket", "dress", "sweater")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print an audit report for the ranking evaluation dataset."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to data/processed/ranking_eval_dataset.parquet.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional manifest path. Defaults next to the dataset parquet.",
    )
    return parser.parse_args()


def normalize_path(path: Path) -> Path:
    return path.resolve()


def derive_manifest_path(dataset_path: Path, manifest_path: Path | None) -> Path:
    if manifest_path is not None:
        return normalize_path(manifest_path)
    return dataset_path.with_name(f"{dataset_path.stem}_manifest.json")


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    rank = max(1, math.ceil(q * len(sorted_values)))
    return float(sorted_values[rank - 1])


def print_numeric_summary(label: str, values: list[float], decimals: int) -> None:
    if not values:
        print(f"  {label}: (none)")
        return
    print(f"  {label}:")
    print(f"    min: {min(values):.{decimals}f}")
    print(f"    mean: {statistics.fmean(values):.{decimals}f}")
    print(f"    median: {statistics.median(values):.{decimals}f}")
    print(f"    p90: {percentile(values, 0.90):.{decimals}f}")
    print(f"    max: {max(values):.{decimals}f}")


def print_count_block(title: str, counts) -> None:
    print(f"  {title}:")
    printed = False
    for name, count in counts:
        printed = True
        print(f"    {name}: {count}")
    if not printed:
        print("    (none)")


def print_tops_diagnostics(frame) -> None:
    tops_frame = frame.loc[frame["ranking_group"] == "tops"]
    print("  tops_fine_label_distribution:")
    if tops_frame.empty:
        print("    (none)")
    else:
        for subcategory, count in tops_frame["subcategory"].value_counts().items():
            print(f"    {subcategory}: {count}")
    print(f"  tops_unknown_rows: {int((tops_frame['subcategory'] == 'unknown').sum()) if not tops_frame.empty else 0}")


def main() -> int:
    args = parse_args()
    dataset_path = normalize_path(args.dataset_path)
    manifest_path = derive_manifest_path(dataset_path, args.manifest_path)

    if not dataset_path.exists():
        print(f"Dataset file does not exist: {dataset_path}", file=sys.stderr)
        return 1

    try:
        import pandas as pd
    except ImportError as exc:
        print(
            "pandas is required to report on the ranking evaluation dataset.\n"
            f"Import error: {exc}",
            file=sys.stderr,
        )
        return 1

    try:
        dataset = pd.read_parquet(dataset_path)
    except Exception as exc:
        print(f"Failed to read ranking dataset: {exc}", file=sys.stderr)
        return 1

    manifest = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"Failed to parse manifest JSON: {exc}", file=sys.stderr)
            return 1

    print()
    print("Ranking evaluation dataset report")
    print(f"Dataset path: {dataset_path}")
    if manifest is not None:
        print(f"Manifest path: {manifest_path}")

    print()
    print("Dataset coverage")
    print(f"  total_rows: {len(dataset)}")
    print(f"  unique_asins: {dataset['asin'].nunique()}")
    print(f"  splits: {', '.join(dataset['split'].drop_duplicates().tolist())}")
    print_count_block(
        "row_counts_by_ranking_group",
        dataset["ranking_group"].value_counts().items(),
    )
    print_count_block(
        "row_counts_by_subcategory",
        dataset["subcategory"].value_counts().items(),
    )
    print_count_block(
        "core_subcategory_counts",
        (
            (subcategory, int((dataset["subcategory"] == subcategory).sum()))
            for subcategory in CORE_SUBCATEGORIES
        ),
    )
    print_tops_diagnostics(dataset)

    if manifest is not None:
        print()
        print("Build filters")
        print(f"  input_reviews: {manifest['input_reviews']}")
        print(
            "  dropped_missing_timestamp_reviews: "
            f"{manifest['dropped_missing_timestamp_reviews']}"
        )
        print(
            "  reviews_after_timestamp_filter: "
            f"{manifest['reviews_after_timestamp_filter']}"
        )
        print(
            "  reviews_after_is_apparel_filter: "
            f"{manifest['reviews_after_is_apparel_filter']}"
        )
        print(
            "  reviews_after_is_apparel_filter_pct: "
            f"{(manifest['reviews_after_is_apparel_filter'] / manifest['reviews_after_timestamp_filter'] * 100.0) if manifest['reviews_after_timestamp_filter'] else 0.0:.2f}%"
        )
        print(
            "  reviews_excluded_non_apparel: "
            f"{manifest['reviews_excluded_non_apparel']}"
        )
        print(
            "  reviews_after_ranking_group_filter: "
            f"{manifest['reviews_after_ranking_group_filter']}"
        )
        print(
            "  reviews_after_ranking_group_filter_pct: "
            f"{(manifest['reviews_after_ranking_group_filter'] / manifest['reviews_after_timestamp_filter'] * 100.0) if manifest['reviews_after_timestamp_filter'] else 0.0:.2f}%"
        )
        print(
            "  reviews_excluded_unknown_ranking_group: "
            f"{manifest['reviews_excluded_unknown_ranking_group']}"
        )

        print()
        print("Split cutoffs")
        for label, payload in manifest["split_cutoffs"].items():
            print(f"  {label}: {payload['timestamp']} ({payload['iso_utc']})")

    for split_name in ("train", "validation", "test"):
        split_frame = dataset.loc[dataset["split"] == split_name]
        print()
        print(f"{split_name.title()} split")
        print(f"  rows: {len(split_frame)}")
        print(f"  unique_asins: {split_frame['asin'].nunique()}")
        print(f"  ranking_groups: {split_frame['ranking_group'].nunique()}")
        print(f"  subcategories: {split_frame['subcategory'].nunique()}")
        print_numeric_summary(
            "target_score_summary",
            split_frame["target_score"].astype(float).tolist(),
            decimals=4,
        )
        print_numeric_summary(
            "future_review_count_summary",
            split_frame["future_review_count"].astype(float).tolist(),
            decimals=2,
        )
        print_count_block(
            "row_counts_by_ranking_group",
            split_frame["ranking_group"].value_counts().items()
            if not split_frame.empty
            else [],
        )
        print_count_block(
            "row_counts_by_subcategory",
            split_frame["subcategory"].value_counts().items()
            if not split_frame.empty
            else [],
        )
        print_count_block(
            "core_subcategory_counts",
            (
                (subcategory, int((split_frame["subcategory"] == subcategory).sum()))
                for subcategory in CORE_SUBCATEGORIES
            ),
        )
        print_tops_diagnostics(split_frame)

        if manifest is not None:
            split_manifest = manifest["splits"].get(split_name, {})
            print_count_block(
                "eligible_ranking_groups",
                split_manifest.get("eligible_ranking_groups", {}).items(),
            )
            print_count_block(
                "excluded_ranking_groups",
                split_manifest.get("excluded_ranking_groups", {}).items(),
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
