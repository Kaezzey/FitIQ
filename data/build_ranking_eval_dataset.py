from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


DEFAULT_REVIEWS_PATH = Path("data/processed/reviews_canonical.parquet")
DEFAULT_TAXONOMY_PATH = Path("data/processed/product_taxonomy.parquet")
DEFAULT_OUTPUT_PATH = Path("data/processed/ranking_eval_dataset.parquet")
UNKNOWN_SUBCATEGORY = "unknown"
UNKNOWN_RANKING_GROUP = "unknown"
MIN_FUTURE_REVIEWS = 2
MIN_GROUP_PRODUCTS = 25
NEGATIVE_FIT_KEYWORDS = (
    "too small",
    "too big",
    "too large",
    "runs small",
    "runs large",
    "run small",
    "run large",
    "tight",
    "too tight",
    "very tight",
    "loose",
    "too loose",
    "very loose",
    "way too small",
    "way too big",
    "way too large",
    "didn't fit",
    "doesn't fit",
    "did not fit",
    "does not fit",
    "poor fit",
    "bad fit",
    "fit issue",
    "size issue",
    "sizing issue",
    "wrong size",
    "size up",
    "sized up",
    "size down",
    "sized down",
    "order a size up",
    "order a size down",
    "go a size up",
    "go a size down",
    "one size up",
    "one size down",
    "half size up",
    "half size down",
    "not true to size",
)
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class SplitSpec:
    name: str
    history_end: int
    label_start_exclusive: int
    label_end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a leakage-safe offline ranking evaluation dataset."
    )
    parser.add_argument(
        "--reviews-path",
        type=Path,
        default=DEFAULT_REVIEWS_PATH,
        help="Path to data/processed/reviews_canonical.parquet.",
    )
    parser.add_argument(
        "--taxonomy-path",
        type=Path,
        default=DEFAULT_TAXONOMY_PATH,
        help="Path to data/processed/product_taxonomy.parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination parquet path for the ranking evaluation dataset.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=None,
        help="Optional JSON manifest path. Defaults next to the parquet output.",
    )
    parser.add_argument(
        "--min-future-reviews",
        type=int,
        default=MIN_FUTURE_REVIEWS,
        help="Minimum holdout-window review count required to emit a label row.",
    )
    parser.add_argument(
        "--min-group-products",
        type=int,
        default=MIN_GROUP_PRODUCTS,
        help="Minimum labeled products required to keep a split/ranking_group group.",
    )
    return parser.parse_args()


def normalize_path(path: Path) -> Path:
    return path.resolve()


def normalize_match_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    normalized = NON_ALNUM_RE.sub(" ", text)
    return WHITESPACE_RE.sub(" ", normalized).strip()


def build_keyword_pattern(keywords: tuple[str, ...]) -> re.Pattern[str]:
    normalized_keywords: list[str] = []
    for keyword in keywords:
        normalized_keyword = normalize_match_text(keyword)
        if normalized_keyword:
            escaped = re.escape(normalized_keyword).replace(r"\ ", r"\s+")
            normalized_keywords.append(escaped)

    normalized_keywords = sorted(set(normalized_keywords), key=len, reverse=True)
    joined = "|".join(normalized_keywords)
    return re.compile(rf"(?<![a-z0-9])(?:{joined})(?![a-z0-9])")


def validate_required_columns(frame, required_columns: set[str], name: str) -> None:
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def validate_one_value_per_asin(reviews, column: str) -> None:
    non_null_counts = reviews.groupby("asin")[column].nunique(dropna=True)
    conflicting = non_null_counts[non_null_counts > 1]
    if conflicting.empty:
        return
    sample = ", ".join(conflicting.index.astype(str).tolist()[:5])
    raise ValueError(
        f"Each ASIN must map to at most one non-null {column}. Conflicting ASINs: {sample}"
    )


def format_timestamp_ms(timestamp_ms: int | None) -> str:
    if timestamp_ms is None:
        return "(none)"
    dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=UTC)
    return dt.isoformat()


def summary_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    sorted_values = sorted(values)
    rank = max(1, math.ceil(0.90 * len(sorted_values)))
    return {
        "min": float(sorted_values[0]),
        "mean": float(statistics.fmean(sorted_values)),
        "median": float(statistics.median(sorted_values)),
        "p90": float(sorted_values[rank - 1]),
        "max": float(sorted_values[-1]),
    }


def derive_manifest_path(output_path: Path, manifest_output: Path | None) -> Path:
    if manifest_output is not None:
        return normalize_path(manifest_output)
    return output_path.with_name(f"{output_path.stem}_manifest.json")


def build_split_specs(reviews) -> list[SplitSpec]:
    timestamps = reviews["timestamp"]
    t60 = int(timestamps.quantile(0.60, interpolation="higher"))
    t75 = int(timestamps.quantile(0.75, interpolation="higher"))
    t90 = int(timestamps.quantile(0.90, interpolation="higher"))
    max_timestamp = int(timestamps.max())
    return [
        SplitSpec(
            name="train",
            history_end=t60,
            label_start_exclusive=t60,
            label_end=t75,
        ),
        SplitSpec(
            name="validation",
            history_end=t75,
            label_start_exclusive=t75,
            label_end=t90,
        ),
        SplitSpec(
            name="test",
            history_end=t90,
            label_start_exclusive=t90,
            label_end=max_timestamp,
        ),
    ]


def aggregate_history(reviews):
    grouped = reviews.groupby("asin", sort=False)
    history = grouped.agg(
        parent_asin=("parent_asin", "first"),
        category=("category", "first"),
        subcategory=("subcategory", "first"),
        ranking_group=("ranking_group", "first"),
        is_apparel=("is_apparel", "first"),
        mean_rating=("rating", "mean"),
        review_count=("rating", "size"),
        fit_complaint_rate=("is_negative_fit_complaint", "mean"),
    )
    history["rating_variance"] = grouped["rating"].var(ddof=0).astype(float)
    history = history.reset_index()
    history["rating_variance"] = history["rating_variance"].fillna(0.0).astype(float)
    history["mean_rating"] = history["mean_rating"].astype(float)
    history["review_count"] = history["review_count"].astype(int)
    history["log_review_count"] = history["review_count"].map(math.log1p).astype(float)
    history["fit_complaint_rate"] = history["fit_complaint_rate"].astype(float)
    return history


def aggregate_future(reviews):
    grouped = reviews.groupby("asin", sort=False)
    future = grouped.agg(
        future_mean_rating=("rating", "mean"),
        future_review_count=("rating", "size"),
        future_fit_complaint_rate=("is_negative_fit_complaint", "mean"),
    )
    future = future.reset_index()
    future["future_mean_rating"] = future["future_mean_rating"].astype(float)
    future["future_review_count"] = future["future_review_count"].astype(int)
    future["future_fit_complaint_rate"] = future["future_fit_complaint_rate"].astype(float)
    return future


def empty_split_frame():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas must be available before building empty split frames.") from exc

    return pd.DataFrame(
        columns=[
            "split",
            "asin",
            "parent_asin",
            "category",
            "subcategory",
            "ranking_group",
            "is_apparel",
            "history_end_timestamp",
            "label_start_exclusive_timestamp",
            "label_end_timestamp",
            "mean_rating",
            "rating_variance",
            "review_count",
            "log_review_count",
            "fit_complaint_rate",
            "future_mean_rating",
            "future_fit_complaint_rate",
            "future_review_count",
            "target_score",
        ]
    )


def build_split_frame(
    reviews,
    split_spec: SplitSpec,
    min_future_reviews: int,
    min_group_products: int,
):
    history_reviews = reviews.loc[reviews["timestamp"] <= split_spec.history_end]
    label_reviews = reviews.loc[
        (reviews["timestamp"] > split_spec.label_start_exclusive)
        & (reviews["timestamp"] <= split_spec.label_end)
    ]

    if not history_reviews.empty and int(history_reviews["timestamp"].max()) > split_spec.history_end:
        raise ValueError(f"History leakage detected in split {split_spec.name}.")
    if not label_reviews.empty:
        if int(label_reviews["timestamp"].min()) <= split_spec.label_start_exclusive:
            raise ValueError(f"Label window start leakage detected in split {split_spec.name}.")
        if int(label_reviews["timestamp"].max()) > split_spec.label_end:
            raise ValueError(f"Label window end leakage detected in split {split_spec.name}.")

    history = aggregate_history(history_reviews)
    future = aggregate_future(label_reviews)

    if history.empty or future.empty:
        return empty_split_frame(), {
            "history_review_rows": int(len(history_reviews)),
            "label_review_rows": int(len(label_reviews)),
            "products_before_label_filter": 0,
            "products_after_label_filter": 0,
            "output_rows": 0,
            "eligible_ranking_groups": {},
            "excluded_ranking_groups": {},
        }

    future = future.loc[future["future_review_count"] >= min_future_reviews].copy()
    combined = history.merge(future, on="asin", how="inner")
    if combined.empty:
        return empty_split_frame(), {
            "history_review_rows": int(len(history_reviews)),
            "label_review_rows": int(len(label_reviews)),
            "products_before_label_filter": int(len(history)),
            "products_after_label_filter": 0,
            "output_rows": 0,
            "eligible_ranking_groups": {},
            "excluded_ranking_groups": {},
        }

    combined["target_score"] = (
        0.5 * ((combined["future_mean_rating"] - 1.0) / 4.0)
        + 0.5 * (1.0 - combined["future_fit_complaint_rate"])
    ).astype(float)

    products_before_group_filter = len(combined)
    ranking_group_sizes = combined.groupby("ranking_group")["asin"].size().sort_values(
        ascending=False
    )
    eligible_ranking_groups = ranking_group_sizes[
        ranking_group_sizes >= min_group_products
    ]
    excluded_ranking_groups = ranking_group_sizes[
        ranking_group_sizes < min_group_products
    ]
    combined = combined.loc[
        combined["ranking_group"].isin(eligible_ranking_groups.index)
    ].copy()

    combined["split"] = split_spec.name
    combined["history_end_timestamp"] = split_spec.history_end
    combined["label_start_exclusive_timestamp"] = split_spec.label_start_exclusive
    combined["label_end_timestamp"] = split_spec.label_end

    audit = {
        "history_review_rows": int(len(history_reviews)),
        "label_review_rows": int(len(label_reviews)),
        "products_before_label_filter": int(len(history)),
        "products_after_label_filter": int(products_before_group_filter),
        "output_rows": int(len(combined)),
        "eligible_ranking_groups": {
            key: int(value) for key, value in eligible_ranking_groups.items()
        },
        "excluded_ranking_groups": {
            key: int(value) for key, value in excluded_ranking_groups.items()
        },
    }
    return combined, audit


def print_audit_summary(
    dataset,
    manifest: dict[str, object],
    output_path: Path,
    manifest_path: Path,
) -> None:
    print()
    print("Ranking evaluation dataset build complete")
    print(f"Output path: {output_path}")
    print(f"Manifest path: {manifest_path}")

    print()
    print("Input filtering")
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
        "  reviews_excluded_non_apparel: "
        f"{manifest['reviews_excluded_non_apparel']}"
    )
    print(
        "  reviews_after_ranking_group_filter: "
        f"{manifest['reviews_after_ranking_group_filter']}"
    )
    print(
        "  reviews_excluded_unknown_ranking_group: "
        f"{manifest['reviews_excluded_unknown_ranking_group']}"
    )

    print()
    print("Split cutoffs")
    for label, payload in manifest["split_cutoffs"].items():
        print(f"  {label}: {payload['timestamp']} ({payload['iso_utc']})")

    print()
    print("Output coverage")
    print(f"  total_rows: {len(dataset)}")
    print(f"  unique_split_asin_rows: {len(dataset)}")
    print(f"  unique_asins: {dataset['asin'].nunique()}")

    for split_name in ("train", "validation", "test"):
        split_frame = dataset.loc[dataset["split"] == split_name]
        split_manifest = manifest["splits"][split_name]
        print()
        print(f"{split_name.title()} split")
        print(f"  rows: {len(split_frame)}")
        print(f"  unique_asins: {split_frame['asin'].nunique()}")
        print(f"  ranking_groups: {split_frame['ranking_group'].nunique()}")
        print(f"  subcategories: {split_frame['subcategory'].nunique()}")
        print(f"  history_review_rows: {split_manifest['history_review_rows']}")
        print(f"  label_review_rows: {split_manifest['label_review_rows']}")

        target_stats = summary_stats(split_frame["target_score"].astype(float).tolist())
        future_count_stats = summary_stats(
            split_frame["future_review_count"].astype(float).tolist()
        )
        print("  target_score_summary:")
        print(f"    min: {target_stats['min']:.4f}")
        print(f"    mean: {target_stats['mean']:.4f}")
        print(f"    median: {target_stats['median']:.4f}")
        print(f"    p90: {target_stats['p90']:.4f}")
        print(f"    max: {target_stats['max']:.4f}")
        print("  future_review_count_summary:")
        print(f"    min: {future_count_stats['min']:.0f}")
        print(f"    mean: {future_count_stats['mean']:.2f}")
        print(f"    median: {future_count_stats['median']:.0f}")
        print(f"    p90: {future_count_stats['p90']:.0f}")
        print(f"    max: {future_count_stats['max']:.0f}")

        print("  eligible_ranking_groups:")
        for ranking_group, count in split_manifest["eligible_ranking_groups"].items():
            print(f"    {ranking_group}: {count}")
        if not split_manifest["eligible_ranking_groups"]:
            print("    (none)")

        print("  excluded_ranking_groups:")
        for ranking_group, count in split_manifest["excluded_ranking_groups"].items():
            print(f"    {ranking_group}: {count}")
        if not split_manifest["excluded_ranking_groups"]:
            print("    (none)")


def main() -> int:
    args = parse_args()
    if args.min_future_reviews < 1:
        print("--min-future-reviews must be at least 1.", file=sys.stderr)
        return 1
    if args.min_group_products < 1:
        print("--min-group-products must be at least 1.", file=sys.stderr)
        return 1

    reviews_path = normalize_path(args.reviews_path)
    taxonomy_path = normalize_path(args.taxonomy_path)
    output_path = normalize_path(args.output)
    manifest_path = derive_manifest_path(output_path, args.manifest_output)

    for path in (reviews_path, taxonomy_path):
        if not path.exists():
            print(f"Input file does not exist: {path}", file=sys.stderr)
            return 1

    try:
        import pandas as pd
    except ImportError as exc:
        print(
            "pandas is required to build the ranking evaluation dataset. Install it "
            "in the Python environment used for this project, then rerun the script.\n"
            f"Import error: {exc}",
            file=sys.stderr,
        )
        return 1

    try:
        print("Loading parquet inputs...")
        reviews = pd.read_parquet(
            reviews_path,
            columns=[
                "asin",
                "parent_asin",
                "category",
                "rating",
                "review_text",
                "timestamp",
            ],
        )
        taxonomy = pd.read_parquet(
            taxonomy_path,
            columns=["parent_asin", "subcategory", "ranking_group", "is_apparel"],
        )
        print(
            f"Loaded {len(reviews):,} review rows and {len(taxonomy):,} taxonomy rows."
        )
    except Exception as exc:
        print(f"Failed to load parquet inputs: {exc}", file=sys.stderr)
        return 1

    validate_required_columns(
        reviews,
        {"asin", "parent_asin", "category", "rating", "review_text", "timestamp"},
        str(reviews_path),
    )
    validate_required_columns(
        taxonomy,
        {"parent_asin", "subcategory", "ranking_group", "is_apparel"},
        str(taxonomy_path),
    )

    if taxonomy["parent_asin"].duplicated().any():
        duplicate_rows = taxonomy.loc[taxonomy["parent_asin"].duplicated(), "parent_asin"]
        sample = ", ".join(duplicate_rows.astype(str).head(5).tolist())
        raise ValueError(f"Taxonomy contains duplicate parent_asin values: {sample}")

    reviews = reviews.copy()
    input_review_count = int(len(reviews))
    reviews["timestamp"] = pd.to_numeric(reviews["timestamp"], errors="coerce")
    dropped_missing_timestamp_reviews = int(reviews["timestamp"].isna().sum())
    reviews = reviews.loc[reviews["timestamp"].notna()].copy()
    reviews["timestamp"] = reviews["timestamp"].astype("int64")
    reviews_after_timestamp_filter = int(len(reviews))

    if reviews.empty:
        raise ValueError("No reviews remain after dropping rows with missing timestamps.")

    print("Joining taxonomy and filtering to apparel ranking groups...")
    reviews = reviews.merge(taxonomy, on="parent_asin", how="left")
    reviews["subcategory"] = reviews["subcategory"].fillna(UNKNOWN_SUBCATEGORY)
    reviews["ranking_group"] = reviews["ranking_group"].fillna(UNKNOWN_RANKING_GROUP)
    reviews["is_apparel"] = reviews["is_apparel"].fillna(False).astype(bool)
    reviews_excluded_non_apparel = int((~reviews["is_apparel"]).sum())
    reviews = reviews.loc[reviews["is_apparel"]].copy()
    reviews_after_is_apparel_filter = int(len(reviews))
    reviews_excluded_unknown_ranking_group = int(
        (reviews["ranking_group"] == UNKNOWN_RANKING_GROUP).sum()
    )
    reviews = reviews.loc[reviews["ranking_group"] != UNKNOWN_RANKING_GROUP].copy()
    reviews_after_ranking_group_filter = int(len(reviews))

    if reviews.empty:
        raise ValueError("No reviews remain after filtering to apparel ranking groups.")

    print("Deriving negative fit complaint flag...")
    negative_fit_pattern = build_keyword_pattern(NEGATIVE_FIT_KEYWORDS)
    normalized_review_text = (
        reviews["review_text"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(NON_ALNUM_RE, " ", regex=True)
        .str.replace(WHITESPACE_RE, " ", regex=True)
        .str.strip()
    )
    reviews["is_negative_fit_complaint"] = normalized_review_text.str.contains(
        negative_fit_pattern,
        regex=True,
        na=False,
    )
    reviews = reviews.drop(columns=["review_text"])

    print("Validating ASIN mappings...")
    validate_one_value_per_asin(reviews, "parent_asin")
    validate_one_value_per_asin(reviews, "category")
    validate_one_value_per_asin(reviews, "subcategory")
    validate_one_value_per_asin(reviews, "ranking_group")

    # Amazon Reviews 2023 review timestamps are stored as Unix epoch milliseconds.
    # We keep split logic on the raw integers and only format them for auditing.
    split_specs = build_split_specs(reviews)
    split_cutoffs = {
        "t60": {
            "timestamp": split_specs[0].history_end,
            "iso_utc": format_timestamp_ms(split_specs[0].history_end),
        },
        "t75": {
            "timestamp": split_specs[1].history_end,
            "iso_utc": format_timestamp_ms(split_specs[1].history_end),
        },
        "t90": {
            "timestamp": split_specs[2].history_end,
            "iso_utc": format_timestamp_ms(split_specs[2].history_end),
        },
        "max_timestamp": {
            "timestamp": split_specs[2].label_end,
            "iso_utc": format_timestamp_ms(split_specs[2].label_end),
        },
    }

    split_frames = []
    split_manifest: dict[str, object] = {}
    for split_spec in split_specs:
        print(f"Building {split_spec.name} split...")
        split_frame, split_audit = build_split_frame(
            reviews=reviews,
            split_spec=split_spec,
            min_future_reviews=args.min_future_reviews,
            min_group_products=args.min_group_products,
        )
        split_frames.append(split_frame)
        split_manifest[split_spec.name] = {
            "history_end_timestamp": split_spec.history_end,
            "history_end_iso_utc": format_timestamp_ms(split_spec.history_end),
            "label_start_exclusive_timestamp": split_spec.label_start_exclusive,
            "label_start_exclusive_iso_utc": format_timestamp_ms(
                split_spec.label_start_exclusive
            ),
            "label_end_timestamp": split_spec.label_end,
            "label_end_iso_utc": format_timestamp_ms(split_spec.label_end),
            **split_audit,
        }

    dataset = pd.concat(split_frames, ignore_index=True)
    if dataset.empty:
        raise ValueError("No ranking evaluation rows were produced.")

    output_columns = [
        "split",
        "asin",
        "parent_asin",
        "category",
        "subcategory",
        "ranking_group",
        "is_apparel",
        "history_end_timestamp",
        "label_start_exclusive_timestamp",
        "label_end_timestamp",
        "mean_rating",
        "rating_variance",
        "review_count",
        "log_review_count",
        "fit_complaint_rate",
        "future_mean_rating",
        "future_fit_complaint_rate",
        "future_review_count",
        "target_score",
    ]
    dataset = dataset.loc[:, output_columns].copy()

    split_order = {"train": 0, "validation": 1, "test": 2}
    dataset["split_order"] = dataset["split"].map(split_order)
    dataset = dataset.sort_values(
        ["split_order", "ranking_group", "subcategory", "asin"], kind="stable"
    ).drop(columns=["split_order"])
    dataset = dataset.reset_index(drop=True)

    if dataset.duplicated(subset=["split", "asin"]).any():
        raise ValueError("Output contains duplicate (split, asin) rows.")
    if (dataset["review_count"] < 1).any():
        raise ValueError("All output rows must have review_count >= 1.")
    if (dataset["future_review_count"] < args.min_future_reviews).any():
        raise ValueError("All output rows must satisfy the minimum future review count.")
    if (dataset["is_apparel"] != True).any():
        raise ValueError("All output rows must have is_apparel = True.")
    if (dataset["ranking_group"] == UNKNOWN_RANKING_GROUP).any():
        raise ValueError("Unknown ranking_group rows must not appear in the output.")

    for column in ("fit_complaint_rate", "future_fit_complaint_rate", "target_score"):
        invalid = (dataset[column] < 0.0) | (dataset[column] > 1.0)
        if invalid.any():
            raise ValueError(f"{column} must stay within [0, 1].")

    group_sizes = dataset.groupby(["split", "ranking_group"])["asin"].size()
    if (group_sizes < args.min_group_products).any():
        raise ValueError("All emitted split/ranking_group groups must meet the size threshold.")

    manifest = {
        "input_reviews": input_review_count,
        "dropped_missing_timestamp_reviews": dropped_missing_timestamp_reviews,
        "reviews_after_timestamp_filter": reviews_after_timestamp_filter,
        "reviews_after_is_apparel_filter": reviews_after_is_apparel_filter,
        "reviews_excluded_non_apparel": reviews_excluded_non_apparel,
        "reviews_after_ranking_group_filter": reviews_after_ranking_group_filter,
        "reviews_excluded_unknown_ranking_group": reviews_excluded_unknown_ranking_group,
        "split_cutoffs": split_cutoffs,
        "splits": split_manifest,
    }

    print("Writing ranking evaluation dataset...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print_audit_summary(
        dataset=dataset,
        manifest=manifest,
        output_path=output_path,
        manifest_path=manifest_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
