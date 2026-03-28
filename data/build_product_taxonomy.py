from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, TextIO


DEFAULT_RAW_ROOT = Path("data/raw")
DEFAULT_OUTPUT_PATH = Path("data/processed/product_taxonomy.parquet")
DEFAULT_CATEGORY_STEMS = ("Amazon_Fashion",)
SUPPORTED_SUFFIXES = (".jsonl", ".json", ".jsonl.gz", ".json.gz")
UNKNOWN_SUBCATEGORY = "unknown"
UNKNOWN_TITLE_SAMPLE_LIMIT = 10

# Keep the taxonomy intentionally small and stable for the first milestone. We
# want within-category ranking groups that are broad enough to have coverage,
# without creating a noisy long tail of labels.
SUBCATEGORY_RULES = (
    (
        "jewelry",
        (
            "jewelry",
            "earring",
            "earrings",
            "necklace",
            "necklaces",
            "bracelet",
            "bracelets",
            "ring",
            "rings",
            "pendant",
            "pendants",
            "anklet",
            "anklets",
            "brooch",
            "brooches",
            "choker",
            "chokers",
            "cufflink",
            "cufflinks",
        ),
    ),
    (
        "shoes",
        (
            "shoe",
            "shoes",
            "sneaker",
            "sneakers",
            "boot",
            "boots",
            "sandal",
            "sandals",
            "heel",
            "heels",
            "loafer",
            "loafers",
            "flat",
            "flats",
            "pump",
            "pumps",
            "slipper",
            "slippers",
            "clog",
            "clogs",
            "mule",
            "mules",
            "moccasin",
            "moccasins",
            "espadrille",
            "espadrilles",
            "wedge",
            "wedges",
        ),
    ),
    (
        "socks_hosiery",
        (
            "sock",
            "socks",
            "hosiery",
            "stocking",
            "stockings",
            "tight",
            "tights",
            "pantyhose",
            "compression sleeve",
            "compression sleeves",
            "calf sleeve",
            "calf sleeves",
        ),
    ),
    (
        "underwear_sleepwear",
        (
            "bra",
            "bras",
            "bralette",
            "bralettes",
            "panty",
            "panties",
            "brief",
            "briefs",
            "boxer",
            "boxers",
            "underwear",
            "undershirt",
            "lingerie",
            "shapewear",
            "camisole",
            "camisoles",
            "cami",
            "camis",
            "pajama",
            "pajamas",
            "pyjama",
            "pyjamas",
            "sleepwear",
            "nightgown",
            "nightshirt",
            "robe",
            "robes",
            "bathrobe",
            "swimsuit",
            "swimwear",
            "bikini",
            "bikinis",
            "one piece swimsuit",
            "tankini",
        ),
    ),
    (
        "dress",
        (
            "dress",
            "dresses",
            "gown",
            "gowns",
            "sundress",
            "sundresses",
            "romper",
            "rompers",
            "jumpsuit",
            "jumpsuits",
        ),
    ),
    (
        "skirt",
        (
            "skirt",
            "skirts",
            "skort",
            "skorts",
        ),
    ),
    (
        "pants",
        (
            "pant",
            "pants",
            "trouser",
            "trousers",
            "jean",
            "jeans",
            "legging",
            "leggings",
            "jogger",
            "joggers",
            "chino",
            "chinos",
            "capri",
            "capris",
            "palazzo",
            "palazzos",
            "short",
            "shorts",
            "sweatpant",
            "sweatpants",
            "cargo pant",
            "cargo pants",
            "slack",
            "slacks",
            "overalls",
        ),
    ),
    (
        "jacket",
        (
            "jacket",
            "jackets",
            "coat",
            "coats",
            "blazer",
            "blazers",
            "parka",
            "parkas",
            "windbreaker",
            "windbreakers",
            "bomber",
            "bombers",
            "trench",
            "trenches",
            "vest",
            "vests",
            "poncho",
            "ponchos",
            "fleece jacket",
            "rain jacket",
        ),
    ),
    (
        "sweater",
        (
            "sweater",
            "sweaters",
            "sweatshirt",
            "sweatshirts",
            "hoodie",
            "hoodies",
            "pullover",
            "pullovers",
            "cardigan",
            "cardigans",
            "jumper",
            "jumpers",
            "knit top",
            "knit tops",
        ),
    ),
    (
        "t_shirt",
        (
            "t shirt",
            "t shirts",
            "tshirt",
            "tshirts",
            "tee",
            "tees",
            "graphic tee",
            "graphic tees",
        ),
    ),
    (
        "shirt_top",
        (
            "shirt",
            "shirts",
            "top",
            "tops",
            "blouse",
            "blouses",
            "tunic",
            "tunics",
            "tank top",
            "tank tops",
            "tank",
            "tanks",
            "polo",
            "polos",
            "henley",
            "henleys",
            "crop top",
            "crop tops",
        ),
    ),
    (
        "accessories",
        (
            "bag",
            "bags",
            "handbag",
            "handbags",
            "backpack",
            "backpacks",
            "purse",
            "purses",
            "wallet",
            "wallets",
            "tote",
            "totes",
            "crossbody",
            "clutch",
            "clutches",
            "duffel",
            "duffels",
            "belt",
            "belts",
            "hat",
            "hats",
            "cap",
            "caps",
            "beanie",
            "beanies",
            "scarf",
            "scarves",
            "glove",
            "gloves",
            "mitten",
            "mittens",
            "sunglasses",
            "watch",
            "watches",
            "headband",
            "headbands",
            "tie",
            "ties",
            "bow tie",
            "bow ties",
            "bowtie",
            "bowties",
        ),
    ),
)

NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
WHITESPACE_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a product taxonomy table for Amazon Reviews 2023 apparel/fashion metadata."
        )
    )
    parser.add_argument(
        "--input",
        nargs="*",
        default=None,
        help=(
            "Raw metadata files or directories. If omitted, the script searches under "
            "data/raw for meta_Amazon_Fashion files."
        ),
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="Root directory used for automatic metadata file discovery.",
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
        help="Destination parquet path for the product taxonomy table.",
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


def discover_input_files(
    explicit_inputs: list[str] | None,
    input_root: Path,
    category_stems: list[str],
) -> list[Path]:
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
        candidate_stems = {stem}
        if not stem.startswith("meta_"):
            candidate_stems.add(f"meta_{stem}")
        if not stem.startswith("raw_meta_"):
            candidate_stems.add(f"raw_meta_{stem}")

        for candidate_stem in candidate_stems:
            for suffix in SUPPORTED_SUFFIXES:
                for path in root.rglob(f"{candidate_stem}{suffix}"):
                    if path.is_file():
                        matches.add(path.resolve())
    return sorted(matches)


def build_missing_data_message(root: Path, category_stems: Iterable[str]) -> str:
    expected_names = ", ".join(
        f"meta_{stem}{suffix}" for stem in category_stems for suffix in SUPPORTED_SUFFIXES
    )
    return (
        "No Amazon Reviews 2023 raw metadata files were found.\n"
        f"Searched under: {root}\n"
        "Expected line-delimited JSON metadata files from the official dataset, such as:\n"
        f"  - {root / 'meta_Amazon_Fashion.jsonl'}\n"
        f"  - {root / 'meta_categories' / 'meta_Amazon_Fashion.jsonl'}\n"
        f"  - {root / 'amazon_reviews_2023' / 'raw' / 'meta_categories' / 'meta_Amazon_Fashion.jsonl'}\n"
        "The loader also accepts .jsonl.gz, .json.gz, or direct file paths passed with --input.\n"
        f"Filename patterns searched: {expected_names}"
    )


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8")
    return path.open(mode="r", encoding="utf-8")


def infer_source_category(path: Path) -> str:
    name = path.name
    for suffix in (".jsonl.gz", ".json.gz", ".jsonl", ".json"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    if name.startswith("raw_meta_"):
        name = name[len("raw_meta_") :]
    elif name.startswith("meta_"):
        name = name[len("meta_") :]
    return name


def clean_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_match_text(value: object) -> str:
    cleaned = clean_text(value)
    if cleaned is None:
        return ""
    lowered = cleaned.lower()
    normalized = NON_ALNUM_RE.sub(" ", lowered)
    return WHITESPACE_RE.sub(" ", normalized).strip()


def parse_categories(value: object) -> list[str]:
    categories: list[str] = []

    def walk(item: object) -> None:
        if item is None:
            return
        if isinstance(item, list):
            for child in item:
                walk(child)
            return
        text = clean_text(item)
        if text is not None:
            categories.append(text)

    walk(value)
    return categories


def serialize_categories(categories: list[str]) -> str:
    return json.dumps(categories, ensure_ascii=True)


def contains_any_keyword(text: str, keywords: Iterable[str]) -> bool:
    if not text:
        return False
    padded_text = f" {text} "
    for keyword in keywords:
        normalized_keyword = normalize_match_text(keyword)
        if normalized_keyword and f" {normalized_keyword} " in padded_text:
            return True
    return False


def derive_subcategory(categories: list[str], title: str | None) -> tuple[str, str]:
    category_text = normalize_match_text(" ".join(categories))
    title_text = normalize_match_text(title)

    for label, keywords in SUBCATEGORY_RULES:
        if contains_any_keyword(category_text, keywords):
            return label, "categories"

    for label, keywords in SUBCATEGORY_RULES:
        if contains_any_keyword(title_text, keywords):
            return label, "title"

    return UNKNOWN_SUBCATEGORY, "unknown"


def resolve_field_map(record: dict[str, object], path: Path) -> dict[str, str | None]:
    field_candidates = {
        "parent_asin": ("parent_asin", "parentAsin"),
        "title": ("title",),
        "categories": ("categories",),
        "main_category": ("main_category", "mainCategory"),
    }
    field_map: dict[str, str | None] = {}
    for target_column, candidates in field_candidates.items():
        field_map[target_column] = next((name for name in candidates if name in record), None)

    if field_map["parent_asin"] is None:
        available = ", ".join(sorted(record.keys()))
        raise ValueError(
            f"Unsupported metadata schema in {path}. Missing required source field: parent_asin. "
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


def build_schema(pyarrow_module) -> object:
    pa = pyarrow_module
    return pa.schema(
        [
            pa.field("parent_asin", pa.string()),
            pa.field("main_category", pa.string()),
            pa.field("raw_categories", pa.string()),
            pa.field("title", pa.string()),
            pa.field("subcategory", pa.string()),
            pa.field("subcategory_source", pa.string()),
        ]
    )


def empty_batch() -> dict[str, list[object]]:
    return {
        "parent_asin": [],
        "main_category": [],
        "raw_categories": [],
        "title": [],
        "subcategory": [],
        "subcategory_source": [],
    }


def write_batch(writer, pyarrow_module, schema, batch: dict[str, list[object]]) -> None:
    pa = pyarrow_module
    arrays = []
    for field in schema:
        arrays.append(pa.array(batch[field.name], type=field.type))
    table = pa.Table.from_arrays(arrays, schema=schema)
    writer.write_table(table)


def print_audit_summary(
    total_products: int,
    missing_counts: Counter[str],
    subcategory_counts: Counter[str],
    source_counts: Counter[str],
    dropped_rows: Counter[str],
    input_files: list[Path],
    output_path: Path,
    unknown_title_samples: list[str],
) -> None:
    unknown_count = subcategory_counts[UNKNOWN_SUBCATEGORY]
    unknown_pct = (unknown_count / total_products * 100.0) if total_products else 0.0

    print()
    print("Product taxonomy build complete")
    print(f"Output path: {output_path}")
    print(f"Input files processed: {len(input_files)}")
    for path in input_files:
        print(f"  - {path}")

    print()
    print("Audit summary")
    print(f"  total_products: {total_products}")
    print(f"  unique_parent_asins: {total_products}")

    print()
    print("Missing counts for key columns")
    for column in ("parent_asin", "title", "raw_categories", "subcategory"):
        print(f"  {column}: {missing_counts[column]}")

    print()
    print("Subcategory distribution")
    for subcategory, count in subcategory_counts.most_common():
        print(f"  {subcategory}: {count}")

    print()
    print("Subcategory source distribution")
    for source, count in source_counts.most_common():
        print(f"  {source}: {count}")

    print()
    print("Unknown subcategory coverage")
    print(f"  unknown_products: {unknown_count}")
    print(f"  unknown_product_pct: {unknown_pct:.2f}%")

    print()
    print("Dropped rows by reason")
    if dropped_rows:
        for reason, count in dropped_rows.items():
            print(f"  {reason}: {count}")
    else:
        print("  (none)")

    print()
    print("Sample unknown titles")
    if unknown_title_samples:
        for title in unknown_title_samples:
            print(f"  - {title}")
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

    total_products = 0
    missing_counts: Counter[str] = Counter(
        {
            "parent_asin": 0,
            "title": 0,
            "raw_categories": 0,
            "subcategory": 0,
        }
    )
    subcategory_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    dropped_rows: Counter[str] = Counter()
    seen_parent_asins: set[str] = set()
    unknown_title_samples: list[str] = []

    writer = None
    try:
        writer = pq.ParquetWriter(str(temp_output_path), schema=schema)

        for input_path in input_files:
            inferred_category = infer_source_category(input_path)
            field_map: dict[str, str | None] | None = None

            for record in iter_json_records(input_path):
                if field_map is None:
                    field_map = resolve_field_map(record, input_path)

                parent_asin = clean_text(record.get(field_map["parent_asin"]))
                if parent_asin is None:
                    dropped_rows["missing_parent_asin"] += 1
                    continue
                if parent_asin in seen_parent_asins:
                    dropped_rows["duplicate_parent_asin"] += 1
                    continue

                title = (
                    clean_text(record.get(field_map["title"]))
                    if field_map["title"] is not None
                    else None
                )
                categories = (
                    parse_categories(record.get(field_map["categories"]))
                    if field_map["categories"] is not None
                    else []
                )
                raw_categories = serialize_categories(categories)
                main_category = (
                    clean_text(record.get(field_map["main_category"]))
                    if field_map["main_category"] is not None
                    else inferred_category
                )
                if main_category is None:
                    main_category = inferred_category

                subcategory, subcategory_source = derive_subcategory(categories, title)

                batch["parent_asin"].append(parent_asin)
                batch["main_category"].append(main_category)
                batch["raw_categories"].append(raw_categories)
                batch["title"].append(title)
                batch["subcategory"].append(subcategory)
                batch["subcategory_source"].append(subcategory_source)

                seen_parent_asins.add(parent_asin)
                total_products += 1
                subcategory_counts[subcategory] += 1
                source_counts[subcategory_source] += 1

                if title is None:
                    missing_counts["title"] += 1
                if not categories:
                    missing_counts["raw_categories"] += 1
                if subcategory == UNKNOWN_SUBCATEGORY and title is not None:
                    if len(unknown_title_samples) < UNKNOWN_TITLE_SAMPLE_LIMIT:
                        unknown_title_samples.append(title)

                if len(batch["parent_asin"]) >= args.batch_size:
                    write_batch(writer, pa, schema, batch)
                    batch = empty_batch()

            if field_map is None:
                dropped_rows["empty_input_file"] += 1

        if total_products == 0:
            print("No usable metadata rows were found after canonicalization.", file=sys.stderr)
            return 1

        if batch["parent_asin"]:
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
        total_products=total_products,
        missing_counts=missing_counts,
        subcategory_counts=subcategory_counts,
        source_counts=source_counts,
        dropped_rows=dropped_rows,
        input_files=input_files,
        output_path=output_path,
        unknown_title_samples=unknown_title_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
