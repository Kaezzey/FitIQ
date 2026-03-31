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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build product-level features from canonical reviews and taxonomy."
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
    apparel_count = int(features["is_apparel"].sum())

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
    print("Apparel coverage")
    print(f"  apparel_products: {apparel_count}")
    print(f"  apparel_product_pct: {(apparel_count / total_products * 100.0) if total_products else 0.0:.2f}%")

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
            columns=["asin", "parent_asin", "category", "rating", "review_text"],
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
        {"asin", "parent_asin", "category", "rating", "review_text"},
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

    print("Validating ASIN to parent_asin mappings...")
    non_null_parent_counts = reviews.groupby("asin")["parent_asin"].nunique(dropna=True)
    conflicting_parent_asins = non_null_parent_counts[non_null_parent_counts > 1]
    if not conflicting_parent_asins.empty:
        sample = ", ".join(conflicting_parent_asins.index.astype(str).tolist()[:5])
        raise ValueError(
            "Each ASIN must map to at most one non-null parent_asin. "
            f"Conflicting ASINs: {sample}"
        )

    print("Aggregating product features...")
    grouped = reviews.groupby("asin", sort=False)
    features = grouped.agg(
        parent_asin=("parent_asin", "first"),
        category=("category", "first"),
        mean_rating=("rating", "mean"),
        review_count=("rating", "size"),
        fit_complaint_rate=("is_negative_fit_complaint", "mean"),
    )
    features["rating_variance"] = grouped["rating"].var(ddof=0).astype(float)
    features = features.reset_index()

    if features.empty:
        raise ValueError("No product features were produced from the input reviews table.")

    features["rating_variance"] = features["rating_variance"].fillna(0.0).astype(float)
    features["mean_rating"] = features["mean_rating"].astype(float)
    features["review_count"] = features["review_count"].astype(int)
    features["log_review_count"] = features["review_count"].map(math.log1p).astype(float)
    features["fit_complaint_rate"] = features["fit_complaint_rate"].astype(float)

    print("Joining taxonomy...")
    features = features.merge(taxonomy, on="parent_asin", how="left")
    unmatched_taxonomy_mask = features["parent_asin"].notna() & features["subcategory"].isna()
    unmatched_taxonomy_count = int(unmatched_taxonomy_mask.sum())
    missing_parent_count = int(features["parent_asin"].isna().sum())
    features["subcategory"] = features["subcategory"].fillna(UNKNOWN_SUBCATEGORY)
    features["ranking_group"] = features["ranking_group"].fillna(UNKNOWN_RANKING_GROUP)
    features["is_apparel"] = features["is_apparel"].fillna(False).astype(bool)

    output_columns = [
        "asin",
        "parent_asin",
        "category",
        "subcategory",
        "ranking_group",
        "is_apparel",
        "mean_rating",
        "rating_variance",
        "review_count",
        "log_review_count",
        "fit_complaint_rate",
    ]
    features = features.loc[:, output_columns]

    if features["asin"].duplicated().any():
        raise ValueError("Output contains duplicate ASIN rows.")
    if (features["review_count"] < 1).any():
        raise ValueError("All products must have review_count >= 1.")
    invalid_fit_rates = (features["fit_complaint_rate"] < 0.0) | (
        features["fit_complaint_rate"] > 1.0
    )
    if invalid_fit_rates.any():
        raise ValueError("fit_complaint_rate must stay within [0, 1].")

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
