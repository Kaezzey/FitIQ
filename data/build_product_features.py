from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path


DEFAULT_REVIEWS_PATH = Path("data/processed/reviews_canonical.parquet")
DEFAULT_TAXONOMY_PATH = Path("data/processed/product_taxonomy.parquet")
DEFAULT_OUTPUT_PATH = Path("data/processed/product_features.parquet")
UNKNOWN_SUBCATEGORY = "unknown"
UNKNOWN_RANKING_GROUP = "unknown"
UNKNOWN_QUERY_GROUP = "unknown"
RECENT_WINDOW_DAYS = 180
RECENT_WINDOW_MS = RECENT_WINDOW_DAYS * 24 * 60 * 60 * 1000
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
TOO_SMALL_KEYWORDS = (
    "too small",
    "runs small",
    "run small",
    "way too small",
    "size up",
    "sized up",
    "order a size up",
    "go a size up",
    "one size up",
    "half size up",
)
TOO_LARGE_KEYWORDS = (
    "too big",
    "too large",
    "runs large",
    "run large",
    "way too big",
    "way too large",
    "size down",
    "sized down",
    "order a size down",
    "go a size down",
    "one size down",
    "half size down",
)
TIGHT_KEYWORDS = ("tight", "too tight", "very tight")
LOOSE_KEYWORDS = ("loose", "too loose", "very loose")
QUALITY_COMPLAINT_KEYWORDS = (
    "poor quality",
    "bad quality",
    "quality issue",
    "poorly made",
    "cheaply made",
    "cheap quality",
    "cheap material",
    "fell apart",
    "falling apart",
    "ripped",
    "tore",
    "torn",
    "broke",
    "broken",
    "defective",
    "itchy",
    "scratchy",
    "thin material",
    "see through",
)
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
WHITESPACE_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build full-corpus product features for scoring and ranking from "
            "canonical reviews and product taxonomy."
        )
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
        help="Destination parquet path for the product features table.",
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


def derive_query_group_v2(ranking_group: object, subcategory: object) -> str:
    ranking_group_value = str(ranking_group or "").strip().lower()
    subcategory_value = str(subcategory or "").strip().lower()

    if ranking_group_value == "tops":
        if subcategory_value == "sweater":
            return "tops_sweater"
        if subcategory_value in {"t_shirt", "shirt_top"}:
            return "tops_shirts"
        return "general_items"

    if ranking_group_value == "bottoms":
        if subcategory_value == "pants":
            return "bottoms_pants"
        return "bottoms_other"

    if ranking_group_value == "outerwear":
        return "outerwear"
    if ranking_group_value == "dress":
        return "dress"
    if ranking_group_value == "footwear":
        return "footwear"
    if ranking_group_value == "intimates_sleep":
        return "intimates_sleep"
    return UNKNOWN_QUERY_GROUP


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    rank = max(1, math.ceil(q * len(sorted_values)))
    return float(sorted_values[rank - 1])


def format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}"


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


def print_audit_summary(
    features,
    unmatched_taxonomy_count: int,
    missing_parent_count: int,
    output_path: Path,
) -> None:
    total_products = len(features)
    unique_asins = features["asin"].nunique(dropna=False)
    unmatched_taxonomy_pct = (
        unmatched_taxonomy_count / total_products * 100.0 if total_products else 0.0
    )
    missing_parent_pct = (
        missing_parent_count / total_products * 100.0 if total_products else 0.0
    )
    review_counts = sorted(features["review_count"].astype(float).tolist())
    fit_rates = sorted(features["fit_complaint_rate"].astype(float).tolist())
    subcategory_counts = Counter(features["subcategory"].tolist())
    ranking_group_counts = Counter(features["ranking_group"].tolist())
    query_group_counts = Counter(features["query_group_v2"].tolist())
    apparel_count = int(features["is_apparel"].sum())
    rankable_mask = features["query_group_v2"] != UNKNOWN_QUERY_GROUP
    rankable_count = int(rankable_mask.sum())

    print()
    print("Product features build complete")
    print(f"Output path: {output_path}")

    print()
    print("Audit summary")
    print(f"  total_products: {total_products}")
    print(f"  unique_asins: {unique_asins}")
    print(f"  missing_parent_asin_products: {missing_parent_count}")
    print(f"  missing_parent_asin_pct: {missing_parent_pct:.2f}%")
    print(f"  unmatched_taxonomy_products: {unmatched_taxonomy_count}")
    print(f"  unmatched_taxonomy_pct: {unmatched_taxonomy_pct:.2f}%")

    print()
    print("Subcategory distribution")
    for subcategory, count in subcategory_counts.most_common():
        print(f"  {subcategory}: {count}")

    print()
    print("Ranking group distribution")
    for ranking_group, count in ranking_group_counts.most_common():
        print(f"  {ranking_group}: {count}")

    print()
    print("Query group distribution")
    for query_group, count in query_group_counts.most_common():
        print(f"  {query_group}: {count}")

    print()
    print("Apparel coverage")
    print(f"  apparel_products: {apparel_count}")
    print(
        f"  apparel_product_pct: {(apparel_count / total_products * 100.0) if total_products else 0.0:.2f}%"
    )
    print(f"  rankable_products: {rankable_count}")
    print(
        f"  rankable_product_pct: {(rankable_count / total_products * 100.0) if total_products else 0.0:.2f}%"
    )

    print()
    print("Reviews per product summary")
    print(f"  min: {format_number(review_counts[0])}")
    print(f"  mean: {statistics.fmean(review_counts):.2f}")
    print(f"  median: {format_number(float(statistics.median(review_counts)))}")
    print(f"  p90: {format_number(percentile(review_counts, 0.90))}")
    print(f"  max: {format_number(review_counts[-1])}")

    print()
    print("Fit complaint rate summary")
    print(f"  min: {fit_rates[0]:.4f}")
    print(f"  mean: {statistics.fmean(fit_rates):.4f}")
    print(f"  median: {float(statistics.median(fit_rates)):.4f}")
    print(f"  p90: {percentile(fit_rates, 0.90):.4f}")
    print(f"  max: {fit_rates[-1]:.4f}")

    print()
    print("Unknown subcategory coverage")
    print(f"  unknown_products: {subcategory_counts[UNKNOWN_SUBCATEGORY]}")


def main() -> int:
    args = parse_args()

    reviews_path = normalize_path(args.reviews_path)
    taxonomy_path = normalize_path(args.taxonomy_path)
    output_path = normalize_path(args.output)

    for path in (reviews_path, taxonomy_path):
        if not path.exists():
            print(f"Input file does not exist: {path}", file=sys.stderr)
            return 1

    try:
        import pandas as pd
    except ImportError as exc:
        print(
            "pandas is required to build product features. Install it in the Python "
            "environment used for this project, then rerun the script.\n"
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
                "verified_purchase",
                "helpful_votes",
            ],
        )
        taxonomy = pd.read_parquet(
            taxonomy_path,
            columns=["parent_asin", "subcategory", "ranking_group", "is_apparel"],
        )
        print(f"Loaded {len(reviews):,} review rows and {len(taxonomy):,} taxonomy rows.")
    except Exception as exc:
        print(f"Failed to load parquet inputs: {exc}", file=sys.stderr)
        return 1

    validate_required_columns(
        reviews,
        {
            "asin",
            "parent_asin",
            "category",
            "rating",
            "review_text",
            "timestamp",
            "verified_purchase",
            "helpful_votes",
        },
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
    normalized_review_text = (
        reviews["review_text"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(NON_ALNUM_RE, " ", regex=True)
        .str.replace(WHITESPACE_RE, " ", regex=True)
        .str.strip()
    )

    print("Deriving review-level feature flags...")
    negative_fit_pattern = build_keyword_pattern(NEGATIVE_FIT_KEYWORDS)
    too_small_pattern = build_keyword_pattern(TOO_SMALL_KEYWORDS)
    too_large_pattern = build_keyword_pattern(TOO_LARGE_KEYWORDS)
    tight_pattern = build_keyword_pattern(TIGHT_KEYWORDS)
    loose_pattern = build_keyword_pattern(LOOSE_KEYWORDS)
    quality_pattern = build_keyword_pattern(QUALITY_COMPLAINT_KEYWORDS)

    reviews["is_negative_fit_complaint"] = normalized_review_text.str.contains(
        negative_fit_pattern,
        regex=True,
        na=False,
    )
    reviews["is_too_small"] = normalized_review_text.str.contains(
        too_small_pattern,
        regex=True,
        na=False,
    )
    reviews["is_too_large"] = normalized_review_text.str.contains(
        too_large_pattern,
        regex=True,
        na=False,
    )
    reviews["is_tight"] = normalized_review_text.str.contains(
        tight_pattern,
        regex=True,
        na=False,
    )
    reviews["is_loose"] = normalized_review_text.str.contains(
        loose_pattern,
        regex=True,
        na=False,
    )
    reviews["is_quality_complaint"] = normalized_review_text.str.contains(
        quality_pattern,
        regex=True,
        na=False,
    )
    reviews["is_low_star"] = reviews["rating"].astype(float) <= 2.0
    reviews["is_high_star"] = reviews["rating"].astype(float) >= 4.0
    reviews["verified_purchase_numeric"] = (
        reviews["verified_purchase"].fillna(False).astype(bool).astype(float)
    )
    reviews["helpful_votes_filled"] = (
        pd.to_numeric(reviews["helpful_votes"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .astype(float)
    )
    reviews = reviews.drop(columns=["review_text", "verified_purchase", "helpful_votes"])

    print("Validating ASIN mappings...")
    validate_one_value_per_asin(reviews, "parent_asin")
    validate_one_value_per_asin(reviews, "category")

    print("Aggregating product features...")
    grouped = reviews.groupby("asin", sort=False)
    features = grouped.agg(
        parent_asin=("parent_asin", "first"),
        category=("category", "first"),
        mean_rating=("rating", "mean"),
        review_count=("rating", "size"),
        fit_complaint_rate=("is_negative_fit_complaint", "mean"),
        low_star_rate=("is_low_star", "mean"),
        high_star_rate=("is_high_star", "mean"),
        too_small_rate=("is_too_small", "mean"),
        too_large_rate=("is_too_large", "mean"),
        tight_rate=("is_tight", "mean"),
        loose_rate=("is_loose", "mean"),
        quality_complaint_rate=("is_quality_complaint", "mean"),
        verified_purchase_rate=("verified_purchase_numeric", "mean"),
        helpful_votes_mean=("helpful_votes_filled", "mean"),
        helpful_votes_sum=("helpful_votes_filled", "sum"),
    )
    features["rating_variance"] = grouped["rating"].var(ddof=0).astype(float)
    features = features.reset_index()

    if features.empty:
        raise ValueError("No product features were produced from the input reviews table.")

    features["rating_variance"] = features["rating_variance"].fillna(0.0).astype(float)
    features["mean_rating"] = features["mean_rating"].astype(float)
    features["review_count"] = features["review_count"].astype(int)
    features["log_review_count"] = features["review_count"].map(math.log1p).astype(float)
    for column in (
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
    ):
        features[column] = features[column].astype(float)
    features["helpful_votes_sum"] = features["helpful_votes_sum"].astype(float)
    features["helpful_votes_sum_log"] = features["helpful_votes_sum"].map(math.log1p).astype(float)
    features = features.drop(columns=["helpful_votes_sum"])

    # Anchor recent features to the latest review timestamp in the full corpus.
    max_timestamp = int(reviews["timestamp"].max())
    recent_window_start = max_timestamp - RECENT_WINDOW_MS
    recent = (
        reviews.loc[reviews["timestamp"] > recent_window_start]
        .groupby("asin", sort=False)
        .agg(
            recent_review_count=("rating", "size"),
            recent_mean_rating=("rating", "mean"),
        )
        .reset_index()
    )
    features = features.merge(recent, on="asin", how="left")
    features["recent_review_count"] = features["recent_review_count"].fillna(0).astype(int)
    features["recent_mean_rating"] = (
        features["recent_mean_rating"].fillna(features["mean_rating"]).astype(float)
    )

    print("Joining taxonomy...")
    features = features.merge(taxonomy, on="parent_asin", how="left")
    unmatched_taxonomy_mask = features["parent_asin"].notna() & features["subcategory"].isna()
    unmatched_taxonomy_count = int(unmatched_taxonomy_mask.sum())
    missing_parent_count = int(features["parent_asin"].isna().sum())
    features["subcategory"] = features["subcategory"].fillna(UNKNOWN_SUBCATEGORY)
    features["ranking_group"] = features["ranking_group"].fillna(UNKNOWN_RANKING_GROUP)
    features["is_apparel"] = features["is_apparel"].fillna(False).astype(bool)
    features["query_group_v2"] = [
        derive_query_group_v2(ranking_group=ranking_group, subcategory=subcategory)
        for ranking_group, subcategory in zip(
            features["ranking_group"],
            features["subcategory"],
            strict=True,
        )
    ]

    output_columns = [
        "asin",
        "parent_asin",
        "category",
        "subcategory",
        "ranking_group",
        "query_group_v2",
        "is_apparel",
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
    features = features.loc[:, output_columns]

    if features["asin"].duplicated().any():
        raise ValueError("Output contains duplicate ASIN rows.")
    if (features["review_count"] < 1).any():
        raise ValueError("All products must have review_count >= 1.")
    for column in (
        "fit_complaint_rate",
        "low_star_rate",
        "high_star_rate",
        "too_small_rate",
        "too_large_rate",
        "tight_rate",
        "loose_rate",
        "quality_complaint_rate",
        "verified_purchase_rate",
    ):
        invalid = (features[column] < 0.0) | (features[column] > 1.0)
        if invalid.any():
            raise ValueError(f"{column} must stay within [0, 1].")
    if (features["helpful_votes_mean"] < 0.0).any():
        raise ValueError("helpful_votes_mean must be non-negative.")
    if (features["helpful_votes_sum_log"] < 0.0).any():
        raise ValueError("helpful_votes_sum_log must be non-negative.")
    if (features["recent_review_count"] < 0).any():
        raise ValueError("recent_review_count must be non-negative.")
    if ((features["recent_mean_rating"] < 1.0) | (features["recent_mean_rating"] > 5.0)).any():
        raise ValueError("recent_mean_rating must stay within [1, 5].")

    print("Writing product features parquet...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)

    print_audit_summary(
        features=features,
        unmatched_taxonomy_count=unmatched_taxonomy_count,
        missing_parent_count=missing_parent_count,
        output_path=output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
