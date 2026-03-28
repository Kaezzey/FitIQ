from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, TextIO


DEFAULT_RAW_ROOT = Path("data/raw")
DEFAULT_OUTPUT_PATH = Path("data/processed/reviews_canonical.parquet")
DEFAULT_CATEGORY_STEMS = ("Amazon_Fashion",)
SUPPORTED_SUFFIXES = (".jsonl", ".json", ".jsonl.gz", ".json.gz")
FIT_KEYWORDS = (
    "too small",
    "too big",
    "too large",
    "runs small",
    "runs large",
    "run small",
    "run large",
    "small fit",
    "large fit",
    "tight",
    "too tight",
    "very tight",
    "loose",
    "too loose",
    "very loose",
    "snug",
    "too snug",
    "baggy",
    "too baggy",
    "oversized",
    "way too small",
    "way too big",
    "way too large",
    "fit small",
    "fit big",
    "fit large",
    "fits small",
    "fits big",
    "fits large",
    "fits tight",
    "fits loose",
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
    "true to size",
    "not true to size",
)

# Amazon Reviews 2023 raw review files are documented with snake_case fields, but
# the dataset also ships helper/sample files that use camelCase names. We accept
# both spellings for the same conceptual columns instead of assuming only one.
SOURCE_FIELD_CANDIDATES = {
    "asin": ("asin",),
    "parent_asin": ("parent_asin", "parentAsin"),
    "rating": ("rating",),
    "review_text": ("text", "review_text"),
    "title": ("title",),
    "timestamp": ("timestamp", "sort_timestamp", "sortTimestamp"),
    "verified_purchase": ("verified_purchase", "verifiedPurchase"),
    "helpful_votes": ("helpful_vote", "helpful_votes", "helpfulVotes"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a canonical reviews table for Amazon Reviews 2023 apparel/fashion data."
        )
    )
    parser.add_argument(
        "--input",
        nargs="*",
        default=None,
        help=(
            "Raw review files or directories. If omitted, the script searches under "
            "data/raw for Amazon_Fashion review files."
        ),
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="Root directory used for automatic raw file discovery.",
    )
    parser.add_argument(
        "--category-stems",
        nargs="*",
        default=list(DEFAULT_CATEGORY_STEMS),
        help=(
            "Category filename stems to search for when --input is omitted. "
            "Example: Amazon_Fashion Clothing_Shoes_and_Jewelry"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination parquet path for the canonical reviews table.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Rows to buffer before each parquet write.",
    )
    return parser.parse_args()


def normalize_path(path: Path) -> Path:
    return path.resolve()


def discover_input_files(explicit_inputs: list[str] | None, input_root: Path, category_stems: list[str]) -> list[Path]:
    if explicit_inputs:
        resolved: list[Path] = []
        for raw_value in explicit_inputs:
            candidate = normalize_path(Path(raw_value))
            if candidate.is_dir():
                resolved.extend(discover_files_under_root(candidate, category_stems))
            elif candidate.is_file():
                resolved.append(candidate)
            else:
                raise FileNotFoundError(f"Input path does not exist: {candidate}")
        deduped = sorted({path for path in resolved})
        if not deduped:
            raise FileNotFoundError("No input files matched the supplied --input paths.")
        return deduped

    root = normalize_path(input_root)
    if not root.exists():
        raise FileNotFoundError(build_missing_data_message(root, category_stems))

    matched = discover_files_under_root(root, category_stems)
    if not matched:
        raise FileNotFoundError(build_missing_data_message(root, category_stems))
    return matched


def discover_files_under_root(root: Path, category_stems: Iterable[str]) -> list[Path]:
    matches: set[Path] = set()
    for stem in category_stems:
        for suffix in SUPPORTED_SUFFIXES:
            for path in root.rglob(f"{stem}{suffix}"):
                if path.is_file():
                    matches.add(path.resolve())
            for path in root.rglob(f"raw_review_{stem}{suffix}"):
                if path.is_file():
                    matches.add(path.resolve())
    return sorted(matches)


def build_missing_data_message(root: Path, category_stems: Iterable[str]) -> str:
    expected_names = ", ".join(f"{stem}{suffix}" for stem in category_stems for suffix in SUPPORTED_SUFFIXES)
    return (
        "No Amazon Reviews 2023 raw review files were found.\n"
        f"Searched under: {root}\n"
        "Expected line-delimited JSON review files from the official dataset, such as:\n"
        f"  - {root / 'Amazon_Fashion.jsonl'}\n"
        f"  - {root / 'review_categories' / 'Amazon_Fashion.jsonl'}\n"
        f"  - {root / 'amazon_reviews_2023' / 'raw' / 'review_categories' / 'Amazon_Fashion.jsonl'}\n"
        "The loader also accepts .jsonl.gz, .json.gz, or direct file paths passed with --input.\n"
        f"Filename patterns searched: {expected_names}"
    )


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8")
    return path.open(mode="r", encoding="utf-8")


def infer_category(path: Path) -> str:
    name = path.name
    for suffix in (".jsonl.gz", ".json.gz", ".jsonl", ".json"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    if name.startswith("raw_review_"):
        name = name[len("raw_review_") :]
    return name


def clean_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_rating(value: object) -> float | None:
    if value is None:
        return None
    try:
        rating = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(rating) or rating < 1.0 or rating > 5.0:
        return None
    return rating


def parse_optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def resolve_field_map(record: dict[str, object], path: Path) -> dict[str, str | None]:
    field_map: dict[str, str | None] = {}
    for target_column, candidates in SOURCE_FIELD_CANDIDATES.items():
        field_map[target_column] = next((name for name in candidates if name in record), None)

    required_columns = ("asin", "rating", "review_text")
    missing = [column for column in required_columns if field_map[column] is None]
    if missing:
        available = ", ".join(sorted(record.keys()))
        missing_list = ", ".join(missing)
        raise ValueError(
            f"Unsupported schema in {path}. Missing required source fields: {missing_list}. "
            f"Available fields: {available}"
        )
    return field_map


def iter_json_records(path: Path) -> Iterator[dict[str, object]]:
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected one JSON object per line in {path}, found {type(record).__name__} at line {line_number}."
                )
            yield record


def percentile(sorted_values: list[int], q: float) -> float:
    if not sorted_values:
        return 0.0
    rank = max(1, math.ceil(q * len(sorted_values)))
    return float(sorted_values[rank - 1])


def format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}"


def build_schema(pyarrow_module) -> object:
    pa = pyarrow_module
    return pa.schema(
        [
            pa.field("asin", pa.string()),
            pa.field("parent_asin", pa.string()),
            pa.field("category", pa.string()),
            pa.field("rating", pa.float32()),
            pa.field("review_text", pa.string()),
            pa.field("timestamp", pa.int64()),
            pa.field("verified_purchase", pa.bool_()),
            pa.field("helpful_votes", pa.int64()),
            pa.field("title", pa.string()),
        ]
    )


def empty_batch() -> dict[str, list[object]]:
    return {
        "asin": [],
        "parent_asin": [],
        "category": [],
        "rating": [],
        "review_text": [],
        "timestamp": [],
        "verified_purchase": [],
        "helpful_votes": [],
        "title": [],
    }


def write_batch(writer, pyarrow_module, schema, batch: dict[str, list[object]]) -> None:
    pa = pyarrow_module
    arrays = []
    for field in schema:
        arrays.append(pa.array(batch[field.name], type=field.type))
    table = pa.Table.from_arrays(arrays, schema=schema)
    writer.write_table(table)


def print_audit_summary(
    total_reviews: int,
    asin_review_counts: Counter[str],
    missing_counts: Counter[str],
    rating_distribution: Counter[float],
    category_distribution: Counter[str],
    fit_keyword_hits: int,
    dropped_rows: Counter[str],
    input_files: list[Path],
    output_path: Path,
) -> None:
    review_count_values = sorted(asin_review_counts.values())
    unique_asins = len(asin_review_counts)
    fit_keyword_pct = (fit_keyword_hits / total_reviews * 100.0) if total_reviews else 0.0

    print()
    print("Canonical reviews table build complete")
    print(f"Output path: {output_path}")
    print(f"Input files processed: {len(input_files)}")
    for path in input_files:
        print(f"  - {path}")

    print()
    print("Audit summary")
    print(f"  total_reviews: {total_reviews}")
    print(f"  unique_asins: {unique_asins}")

    print()
    print("Missing counts for key columns")
    for column in ("asin", "parent_asin", "category", "rating", "review_text"):
        print(f"  {column}: {missing_counts[column]}")

    print()
    print("Rating distribution")
    for rating in sorted(rating_distribution):
        print(f"  {format_number(rating)}: {rating_distribution[rating]}")

    print()
    print("Top categories by count")
    if category_distribution:
        for category, count in category_distribution.most_common(10):
            print(f"  {category}: {count}")
    else:
        print("  (none)")

    print()
    print("Reviews per product summary")
    if review_count_values:
        mean_value = statistics.fmean(review_count_values)
        median_value = statistics.median(review_count_values)
        p90_value = percentile(review_count_values, 0.90)
        print(f"  min: {review_count_values[0]}")
        print(f"  mean: {mean_value:.2f}")
        print(f"  median: {format_number(float(median_value))}")
        print(f"  p90: {format_number(p90_value)}")
        print(f"  max: {review_count_values[-1]}")
    else:
        print("  (none)")

    print()
    print("Fit keyword coverage")
    print(f"  keywords: {', '.join(FIT_KEYWORDS)}")
    print(f"  matching_reviews: {fit_keyword_hits}")
    print(f"  matching_review_pct: {fit_keyword_pct:.2f}%")

    print()
    print("Dropped rows by reason")
    if dropped_rows:
        for reason, count in dropped_rows.items():
            print(f"  {reason}: {count}")
    else:
        print("  (none)")


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        print("--batch-size must be a positive integer.", file=sys.stderr)
        return 1

    try:
        input_files = discover_input_files(args.input, args.input_root, args.category_stems)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        print(
            "pyarrow is required to write parquet output. Install it in the Python environment "
            "used for this project, then rerun the script.\n"
            f"Import error: {exc}",
            file=sys.stderr,
        )
        return 1

    output_path = normalize_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")

    schema = build_schema(pa)
    batch = empty_batch()

    total_reviews = 0
    asin_review_counts: Counter[str] = Counter()
    # Required canonical columns are validated before a row is written, so the
    # missing counts in the final parquet should remain zero.
    missing_counts: Counter[str] = Counter(
        {
            "asin": 0,
            "parent_asin": 0,
            "category": 0,
            "rating": 0,
            "review_text": 0,
        }
    )
    rating_distribution: Counter[float] = Counter()
    category_distribution: Counter[str] = Counter()
    dropped_rows: Counter[str] = Counter()
    fit_keyword_hits = 0

    writer = None
    try:
        writer = pq.ParquetWriter(str(temp_output_path), schema=schema)

        for input_path in input_files:
            # Amazon Reviews 2023 review records do not carry a category field, so
            # we attach the category from the source filename.
            category = infer_category(input_path)
            field_map: dict[str, str | None] | None = None

            for record in iter_json_records(input_path):
                if field_map is None:
                    field_map = resolve_field_map(record, input_path)

                asin = clean_text(record.get(field_map["asin"])) if field_map["asin"] else None
                parent_asin = (
                    clean_text(record.get(field_map["parent_asin"]))
                    if field_map["parent_asin"]
                    else None
                )
                rating = parse_rating(record.get(field_map["rating"])) if field_map["rating"] else None
                review_text = (
                    clean_text(record.get(field_map["review_text"]))
                    if field_map["review_text"]
                    else None
                )

                if asin is None:
                    dropped_rows["missing_asin"] += 1
                    continue
                if rating is None:
                    dropped_rows["invalid_or_missing_rating"] += 1
                    continue
                if review_text is None:
                    dropped_rows["missing_review_text"] += 1
                    continue

                timestamp = (
                    parse_optional_int(record.get(field_map["timestamp"]))
                    if field_map["timestamp"]
                    else None
                )
                verified_purchase = (
                    parse_optional_bool(record.get(field_map["verified_purchase"]))
                    if field_map["verified_purchase"]
                    else None
                )
                helpful_votes = (
                    parse_optional_int(record.get(field_map["helpful_votes"]))
                    if field_map["helpful_votes"]
                    else None
                )
                title = clean_text(record.get(field_map["title"])) if field_map["title"] else None

                batch["asin"].append(asin)
                batch["parent_asin"].append(parent_asin)
                batch["category"].append(category)
                batch["rating"].append(rating)
                batch["review_text"].append(review_text)
                batch["timestamp"].append(timestamp)
                batch["verified_purchase"].append(verified_purchase)
                batch["helpful_votes"].append(helpful_votes)
                batch["title"].append(title)

                total_reviews += 1
                asin_review_counts[asin] += 1
                rating_distribution[rating] += 1
                category_distribution[category] += 1
                if parent_asin is None:
                    missing_counts["parent_asin"] += 1
                review_text_lower = review_text.lower()
                if any(keyword in review_text_lower for keyword in FIT_KEYWORDS):
                    fit_keyword_hits += 1

                if len(batch["asin"]) >= args.batch_size:
                    write_batch(writer, pa, schema, batch)
                    batch = empty_batch()

            if field_map is None:
                dropped_rows["empty_input_file"] += 1

        if total_reviews == 0:
            print("No usable reviews were found after canonicalization.", file=sys.stderr)
            return 1

        if batch["asin"]:
            write_batch(writer, pa, schema, batch)

        writer.close()
        writer = None

        if output_path.exists():
            output_path.unlink()
        temp_output_path.replace(output_path)

    finally:
        if writer is not None:
            writer.close()
        if temp_output_path.exists() and not output_path.exists():
            temp_output_path.unlink(missing_ok=True)

    print_audit_summary(
        total_reviews=total_reviews,
        asin_review_counts=asin_review_counts,
        missing_counts=missing_counts,
        rating_distribution=rating_distribution,
        category_distribution=category_distribution,
        fit_keyword_hits=fit_keyword_hits,
        dropped_rows=dropped_rows,
        input_files=input_files,
        output_path=output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
