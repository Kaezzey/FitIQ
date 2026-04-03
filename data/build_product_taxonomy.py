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
UNKNOWN_RANKING_GROUP = "unknown"
UNKNOWN_TITLE_SAMPLE_LIMIT = 10
FIELD_WEIGHTS = {
    "categories": 5.0,
    "title": 4.0,
    "features": 2.5,
    "description": 1.5,
}

# Keep the taxonomy broad and stable enough for within-category ranking, but
# cover more of Amazon Fashion's long tail than title-only apparel buckets.
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
            "ear cuff",
            "ear cuffs",
            "piercing",
            "piercings",
            "barbell",
            "barbells",
            "charm",
            "charms",
            "chain",
            "chains",
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
            "slide",
            "slides",
            "cleat",
            "cleats",
            "bootie",
            "booties",
        ),
    ),
    (
        "eyewear",
        (
            "glasses",
            "glass",
            "eyeglasses",
            "eyeglass",
            "sunglasses",
            "goggles",
            "spectacle",
            "spectacles",
            "optical frame",
            "optical frames",
            "frame",
            "frames",
            "reader",
            "readers",
            "reading glasses",
            "lens",
            "lenses",
            "rimless",
            "horn rim",
        ),
    ),
    (
        "face_covering",
        (
            "face mask",
            "face masks",
            "mask",
            "masks",
            "mouth mask",
            "mouth masks",
            "balaclava",
            "balaclavas",
            "gaiter",
            "gaiters",
            "neck gaiter",
            "neck gaiters",
            "face covering",
            "face coverings",
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
            "lounger",
            "loungers",
            "loungewear",
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
        "uniform_costume",
        (
            "jersey",
            "jerseys",
            "leotard",
            "leotards",
            "unitard",
            "unitards",
            "tutu",
            "tutus",
            "ballet",
            "gymnastics",
            "gymnastic",
            "dancewear",
            "dance",
            "costume",
            "costumes",
            "cosplay",
            "uniform",
            "uniforms",
        ),
    ),
    (
        "baby_kids",
        (
            "sleeper",
            "sleepers",
            "onesie",
            "onesies",
            "bodysuit",
            "bodysuits",
            "booties",
            "bootie",
            "bib",
            "bibs",
            "swaddle",
            "swaddles",
            "baby boy",
            "baby girl",
            "toddler",
            "newborn",
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
            "gilet",
            "soft shell",
            "outerwear",
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
            "keychain",
            "keychains",
            "key ring",
            "key rings",
            "keyring",
            "keyrings",
            "lanyard",
            "lanyards",
            "wristband",
            "wristbands",
            "strap",
            "straps",
            "hair wrap",
            "hair wraps",
            "turban",
            "turbans",
            "pouch",
            "pouches",
            "umbrella",
            "umbrellas",
            "apron",
            "aprons",
            "bandana",
            "bandanas",
            "banner",
            "flags",
        ),
    ),
)

RANKING_GROUP_BY_SUBCATEGORY = {
    "t_shirt": "tops",
    "shirt_top": "tops",
    "sweater": "tops",
    "pants": "bottoms",
    "skirt": "bottoms",
    "jacket": "outerwear",
    "dress": "dress",
    "underwear_sleepwear": "intimates_sleep",
    "shoes": "footwear",
}

TOP_FINE_LABEL_ORDER = ("t_shirt", "shirt_top", "sweater")
SHIRT_TOP_GENERIC_KEYWORDS = ("shirt", "shirts", "top", "tops")
SHIRT_TOP_SPECIFIC_KEYWORDS = (
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
    "button down",
    "button down shirt",
    "button up",
    "button up shirt",
    "button front",
)
TOP_FINE_RULES = {
    "t_shirt": {
        "positive": (
            "t shirt",
            "t shirts",
            "tshirt",
            "tshirts",
            "tee",
            "tees",
            "graphic tee",
            "graphic tees",
            "crew tee",
            "crew tees",
            "crew neck tee",
            "crew neck tees",
            "pocket tee",
            "pocket tees",
            "short sleeve tee",
            "short sleeve tees",
            "v neck tee",
            "v neck tees",
        ),
        "exclude": (
            "polo",
            "polos",
            "blouse",
            "blouses",
            "tunic",
            "tunics",
            "cardigan",
            "cardigans",
            "hoodie",
            "hoodies",
            "sweater",
            "sweaters",
            "sweatshirt",
            "sweatshirts",
            "pullover",
            "pullovers",
        ),
    },
    "shirt_top": {
        "positive": (
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
            "button down",
            "button down shirt",
            "button up",
            "button up shirt",
            "button front",
        ),
        "exclude": (
            "tee",
            "tees",
            "t shirt",
            "t shirts",
            "tshirt",
            "tshirts",
            "graphic tee",
            "graphic tees",
            "hoodie",
            "hoodies",
            "cardigan",
            "cardigans",
            "sweater",
            "sweaters",
            "sweatshirt",
            "sweatshirts",
        ),
    },
    "sweater": {
        "positive": (
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
            "knit",
            "knits",
            "knitted",
            "fleece",
            "thermal knit",
            "chunky knit",
        ),
        "exclude": (
            "tee",
            "tees",
            "t shirt",
            "t shirts",
            "tshirt",
            "tshirts",
            "graphic tee",
            "graphic tees",
            "polo",
            "polos",
            "blouse",
            "blouses",
            "tank top",
            "tank tops",
        ),
    },
}

BROAD_RANKING_GROUP_RULES = (
    (
        "tops",
        (
            "top",
            "tops",
            "shirt",
            "shirts",
            "tee",
            "tees",
            "t shirt",
            "t shirts",
            "tshirt",
            "tshirts",
            "blouse",
            "blouses",
            "tank",
            "tanks",
            "tank top",
            "tank tops",
            "tunic",
            "tunics",
            "polo",
            "polos",
            "henley",
            "henleys",
            "crew neck",
            "crewneck",
            "v neck",
            "vneck",
            "mock neck",
            "long sleeve",
            "short sleeve",
            "sleeveless",
            "sweater",
            "sweaters",
            "hoodie",
            "hoodies",
            "sweatshirt",
            "sweatshirts",
            "cardigan",
            "cardigans",
            "pullover",
            "pullovers",
            "jumper",
            "jumpers",
            "knit",
            "knitted",
            "fleece",
        ),
    ),
    (
        "bottoms",
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
            "short",
            "shorts",
            "skirt",
            "skirts",
            "skort",
            "skorts",
            "waistband",
            "drawstring",
            "inseam",
        ),
    ),
    (
        "outerwear",
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
            "outerwear",
            "full zip",
            "zip front",
            "zip up",
            "shell",
            "puffer",
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
            "maxi dress",
            "midi dress",
            "mini dress",
        ),
    ),
    (
        "intimates_sleep",
        (
            "bra",
            "bras",
            "brief",
            "briefs",
            "panty",
            "panties",
            "underwear",
            "lingerie",
            "sleepwear",
            "nightwear",
            "nightgown",
            "nightshirt",
            "robe",
            "robes",
            "pajama",
            "pajamas",
            "pyjama",
            "pyjamas",
            "swimwear",
            "swimsuit",
            "bikini",
            "bikinis",
            "loungewear",
        ),
    ),
    (
        "footwear",
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
            "slipper",
            "slippers",
            "clog",
            "clogs",
            "mule",
            "mules",
            "wedge",
            "wedges",
            "slide",
            "slides",
            "footwear",
        ),
    ),
)

NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
WHITESPACE_RE = re.compile(r"\s+")
FIELD_CANDIDATES = {
    "parent_asin": ("parent_asin", "parentAsin"),
    "title": ("title",),
    "categories": ("categories",),
    "main_category": ("main_category", "mainCategory"),
    "features": ("features",),
    "description": ("description",),
}


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
        candidate_stems: set[str] = set()
        if stem.startswith("raw_meta_"):
            candidate_stems.add(stem)
        elif stem.startswith("meta_"):
            candidate_stems.add(stem)
            candidate_stems.add(f"raw_meta_{stem}")
        else:
            # Only match metadata files here. The review files share fields like
            # parent_asin and title, so accidentally loading them would silently
            # build a taxonomy from review titles instead of product metadata.
            candidate_stems.add(f"meta_{stem}")
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


def parse_text_list(value: object) -> list[str]:
    texts: list[str] = []

    def walk(item: object) -> None:
        if item is None:
            return
        if isinstance(item, list):
            for child in item:
                walk(child)
            return
        text = clean_text(item)
        if text is not None:
            texts.append(text)

    walk(value)
    return texts


def build_keyword_patterns(
    rules: tuple[tuple[str, tuple[str, ...]], ...]
) -> list[tuple[str, re.Pattern[str]]]:
    patterns: list[tuple[str, re.Pattern[str]]] = []
    for label, keywords in rules:
        normalized_keywords: list[str] = []
        for keyword in keywords:
            normalized_keyword = normalize_match_text(keyword)
            if normalized_keyword:
                escaped = re.escape(normalized_keyword).replace(r"\ ", r"\s+")
                normalized_keywords.append(escaped)
        normalized_keywords = sorted(set(normalized_keywords), key=len, reverse=True)
        joined = "|".join(normalized_keywords)
        patterns.append((label, re.compile(rf"(?<![a-z0-9])(?:{joined})(?![a-z0-9])")))
    return patterns


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


SUBCATEGORY_PATTERNS = build_keyword_patterns(SUBCATEGORY_RULES)
BROAD_RANKING_GROUP_PATTERNS = build_keyword_patterns(BROAD_RANKING_GROUP_RULES)
TOP_FINE_PATTERNS = {
    label: {
        "positive": build_keyword_pattern(rule["positive"]),
        "exclude": build_keyword_pattern(rule["exclude"]),
    }
    for label, rule in TOP_FINE_RULES.items()
}
SHIRT_TOP_GENERIC_PATTERN = build_keyword_pattern(SHIRT_TOP_GENERIC_KEYWORDS)
SHIRT_TOP_SPECIFIC_PATTERN = build_keyword_pattern(SHIRT_TOP_SPECIFIC_KEYWORDS)


def count_pattern_matches(text: str, pattern: re.Pattern[str]) -> int:
    return sum(1 for _ in pattern.finditer(text))


def score_patterns(
    field_texts: dict[str, str],
    patterns: list[tuple[str, re.Pattern[str]]],
) -> tuple[str | None, str | None, dict[str, float]]:
    scores = {label: 0.0 for label, _ in patterns}
    best_field_for_label: dict[str, tuple[float, str]] = {}
    label_order = {label: index for index, (label, _) in enumerate(patterns)}

    for label_index, (label, pattern) in enumerate(patterns):
        best_field_weight = 0.0
        best_field_name: str | None = None
        for field_name, field_text in field_texts.items():
            if not field_text:
                continue
            if pattern.search(field_text):
                field_weight = FIELD_WEIGHTS[field_name]
                scores[label] += field_weight
                if field_weight > best_field_weight:
                    best_field_weight = field_weight
                    best_field_name = field_name
        if best_field_name is not None:
            best_field_for_label[label] = (best_field_weight, best_field_name)

    ranked_labels = [
        label for label, score in scores.items() if score > 0.0
    ]
    if not ranked_labels:
        return None, None, scores

    ranked_labels.sort(
        key=lambda label: (
            -scores[label],
            -best_field_for_label.get(label, (0.0, ""))[0],
            label_order[label],
        )
    )
    best_label = ranked_labels[0]
    best_field_name = best_field_for_label[best_label][1]
    return best_label, best_field_name, scores


def classify_top_fine_subcategory(
    field_texts: dict[str, str],
) -> tuple[str, str]:
    scores = {label: 0.0 for label in TOP_FINE_LABEL_ORDER}
    strong_field_hits = {label: 0 for label in TOP_FINE_LABEL_ORDER}
    weak_field_hits = {label: 0 for label in TOP_FINE_LABEL_ORDER}
    best_positive_field: dict[str, tuple[float, str]] = {}

    for label in TOP_FINE_LABEL_ORDER:
        positive_pattern = TOP_FINE_PATTERNS[label]["positive"]
        exclude_pattern = TOP_FINE_PATTERNS[label]["exclude"]
        strongest_positive_weight = 0.0
        strongest_positive_field: str | None = None
        strong_exclusion_without_strong_positive = False

        for field_name, field_text in field_texts.items():
            if not field_text:
                continue

            positive_count = count_pattern_matches(field_text, positive_pattern)
            exclude_count = count_pattern_matches(field_text, exclude_pattern)

            if positive_count:
                weighted_positive = FIELD_WEIGHTS[field_name] * min(2, positive_count)
                scores[label] += weighted_positive
                if FIELD_WEIGHTS[field_name] > strongest_positive_weight:
                    strongest_positive_weight = FIELD_WEIGHTS[field_name]
                    strongest_positive_field = field_name
                if field_name in {"categories", "title"}:
                    strong_field_hits[label] += 1
                else:
                    weak_field_hits[label] += 1

            if exclude_count:
                weighted_exclusion = FIELD_WEIGHTS[field_name] * min(2, exclude_count)
                if field_name in {"categories", "title"} and strong_field_hits[label] == 0:
                    strong_exclusion_without_strong_positive = True
                scores[label] -= weighted_exclusion

        if strong_exclusion_without_strong_positive:
            scores[label] = 0.0
        if scores[label] < 0.0:
            scores[label] = 0.0
        if strongest_positive_field is not None:
            best_positive_field[label] = (
                strongest_positive_weight,
                strongest_positive_field,
            )

    ranked_labels = sorted(
        TOP_FINE_LABEL_ORDER,
        key=lambda label: (-scores[label], TOP_FINE_LABEL_ORDER.index(label)),
    )
    best_label = ranked_labels[0]
    second_label = ranked_labels[1]
    best_score = scores[best_label]
    second_score = scores[second_label]

    if best_score <= 0.0:
        return UNKNOWN_SUBCATEGORY, "tops_weak_signal"

    has_strong_evidence = strong_field_hits[best_label] > 0
    has_multiple_weak_hits = weak_field_hits[best_label] >= 2
    if not has_strong_evidence and not has_multiple_weak_hits:
        return UNKNOWN_SUBCATEGORY, "tops_weak_signal"

    if best_label == "shirt_top":
        generic_hits = 0
        specific_hits = 0
        for field_text in field_texts.values():
            if not field_text:
                continue
            generic_hits += count_pattern_matches(field_text, SHIRT_TOP_GENERIC_PATTERN)
            specific_hits += count_pattern_matches(field_text, SHIRT_TOP_SPECIFIC_PATTERN)
        if specific_hits == 0 and generic_hits > 0:
            return UNKNOWN_SUBCATEGORY, "tops_generic_only"

    if second_score > 0.0 and (best_score - second_score) < 2.0:
        return UNKNOWN_SUBCATEGORY, "tops_ambiguous"

    if (
        not has_strong_evidence
        and second_score > 0.0
        and (best_score - second_score) < 3.0
    ):
        return UNKNOWN_SUBCATEGORY, "tops_ambiguous"

    source_field = best_positive_field.get(best_label, (0.0, "unknown"))[1]
    return best_label, f"tops_scored_{source_field}"


def derive_subcategory(
    categories: list[str],
    title: str | None,
    features: list[str],
    description: list[str],
) -> tuple[str, str, str, bool]:
    field_texts = {
        "categories": normalize_match_text(" ".join(categories)),
        "title": normalize_match_text(title),
        "features": normalize_match_text(" ".join(features)),
        "description": normalize_match_text(" ".join(description)),
    }

    best_ranking_group, broad_field_name, _ = score_patterns(
        field_texts, BROAD_RANKING_GROUP_PATTERNS
    )
    if best_ranking_group is not None:
        if best_ranking_group == "tops":
            top_subcategory, top_source = classify_top_fine_subcategory(field_texts)
            return top_subcategory, top_source, "tops", True

        best_subcategory, best_field_name, _ = score_patterns(
            field_texts, SUBCATEGORY_PATTERNS
        )
        if best_subcategory is not None:
            mapped_group = RANKING_GROUP_BY_SUBCATEGORY.get(
                best_subcategory, UNKNOWN_RANKING_GROUP
            )
            if mapped_group == best_ranking_group:
                return (
                    best_subcategory,
                    f"scored_{best_field_name}",
                    best_ranking_group,
                    True,
                )
        return (
            UNKNOWN_SUBCATEGORY,
            f"fallback_{broad_field_name}",
            best_ranking_group,
            True,
        )

    best_subcategory, best_field_name, _ = score_patterns(field_texts, SUBCATEGORY_PATTERNS)
    if best_subcategory is not None:
        ranking_group = RANKING_GROUP_BY_SUBCATEGORY.get(
            best_subcategory, UNKNOWN_RANKING_GROUP
        )
        if ranking_group == "tops":
            top_subcategory, top_source = classify_top_fine_subcategory(field_texts)
            return top_subcategory, top_source, "tops", True
        is_apparel = ranking_group != UNKNOWN_RANKING_GROUP
        return best_subcategory, f"scored_{best_field_name}", ranking_group, is_apparel

    return UNKNOWN_SUBCATEGORY, "unknown", UNKNOWN_RANKING_GROUP, False


def resolve_field_map(record: dict[str, object], path: Path) -> dict[str, str | None]:
    field_map: dict[str, str | None] = {}
    for target_column, candidates in FIELD_CANDIDATES.items():
        field_map[target_column] = next((name for name in candidates if name in record), None)

    if field_map["parent_asin"] is None:
        available = ", ".join(sorted(record.keys()))
        raise ValueError(
            f"Unsupported metadata schema in {path}. Missing required source field: parent_asin. "
            f"Available fields: {available}"
        )

    metadata_signal_fields = (
        field_map["categories"],
        field_map["features"],
        field_map["description"],
        field_map["main_category"],
    )
    if not any(metadata_signal_fields):
        available = ", ".join(sorted(record.keys()))
        raise ValueError(
            "Input does not look like Amazon Reviews 2023 metadata. Expected at least one "
            "metadata-specific field such as categories, features, description, or "
            f"main_category in {path}. Available fields: {available}"
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
            pa.field("ranking_group", pa.string()),
            pa.field("is_apparel", pa.bool_()),
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
        "ranking_group": [],
        "is_apparel": [],
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
    ranking_group_counts: Counter[str],
    tops_subcategory_counts: Counter[str],
    apparel_counts: Counter[bool],
    source_counts: Counter[str],
    dropped_rows: Counter[str],
    input_files: list[Path],
    output_path: Path,
    unknown_title_samples: list[str],
    tops_unknown_title_samples: list[str],
) -> None:
    unknown_count = subcategory_counts[UNKNOWN_SUBCATEGORY]
    unknown_pct = (unknown_count / total_products * 100.0) if total_products else 0.0
    apparel_count = apparel_counts[True]
    apparel_pct = (apparel_count / total_products * 100.0) if total_products else 0.0
    tops_unknown_count = tops_subcategory_counts[UNKNOWN_SUBCATEGORY]

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
    for column in ("parent_asin", "title", "raw_categories", "subcategory", "ranking_group"):
        print(f"  {column}: {missing_counts[column]}")

    print()
    print("Subcategory distribution")
    for subcategory, count in subcategory_counts.most_common():
        print(f"  {subcategory}: {count}")

    print()
    print("Ranking group distribution")
    for ranking_group, count in ranking_group_counts.most_common():
        print(f"  {ranking_group}: {count}")

    print()
    print("Subcategory source distribution")
    for source, count in source_counts.most_common():
        print(f"  {source}: {count}")

    print()
    print("Apparel coverage")
    print(f"  apparel_products: {apparel_count}")
    print(f"  apparel_product_pct: {apparel_pct:.2f}%")
    print(f"  non_apparel_products: {apparel_counts[False]}")

    print()
    print("Unknown subcategory coverage")
    print(f"  unknown_products: {unknown_count}")
    print(f"  unknown_product_pct: {unknown_pct:.2f}%")

    print()
    print("Tops fine-label diagnostics")
    for subcategory, count in tops_subcategory_counts.most_common():
        print(f"  {subcategory}: {count}")
    if not tops_subcategory_counts:
        print("  (none)")
    print(f"  tops_unknown_products: {tops_unknown_count}")

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

    print()
    print("Sample tops unknown titles")
    if tops_unknown_title_samples:
        for title in tops_unknown_title_samples:
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
            "ranking_group": 0,
        }
    )
    subcategory_counts: Counter[str] = Counter()
    ranking_group_counts: Counter[str] = Counter()
    tops_subcategory_counts: Counter[str] = Counter()
    apparel_counts: Counter[bool] = Counter()
    source_counts: Counter[str] = Counter()
    dropped_rows: Counter[str] = Counter()
    seen_parent_asins: set[str] = set()
    unknown_title_samples: list[str] = []
    tops_unknown_title_samples: list[str] = []

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
                features = (
                    parse_text_list(record.get(field_map["features"]))
                    if field_map["features"] is not None
                    else []
                )
                description = (
                    parse_text_list(record.get(field_map["description"]))
                    if field_map["description"] is not None
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

                subcategory, subcategory_source, ranking_group, is_apparel = derive_subcategory(
                    categories=categories,
                    title=title,
                    features=features,
                    description=description,
                )

                batch["parent_asin"].append(parent_asin)
                batch["main_category"].append(main_category)
                batch["raw_categories"].append(raw_categories)
                batch["title"].append(title)
                batch["subcategory"].append(subcategory)
                batch["subcategory_source"].append(subcategory_source)
                batch["ranking_group"].append(ranking_group)
                batch["is_apparel"].append(is_apparel)

                seen_parent_asins.add(parent_asin)
                total_products += 1
                subcategory_counts[subcategory] += 1
                ranking_group_counts[ranking_group] += 1
                if ranking_group == "tops":
                    tops_subcategory_counts[subcategory] += 1
                apparel_counts[is_apparel] += 1
                source_counts[subcategory_source] += 1

                if title is None:
                    missing_counts["title"] += 1
                if not categories:
                    missing_counts["raw_categories"] += 1
                if ranking_group == UNKNOWN_RANKING_GROUP:
                    missing_counts["ranking_group"] += 1
                if subcategory == UNKNOWN_SUBCATEGORY and ranking_group == UNKNOWN_RANKING_GROUP and title is not None:
                    if len(unknown_title_samples) < UNKNOWN_TITLE_SAMPLE_LIMIT:
                        unknown_title_samples.append(title)
                if ranking_group == "tops" and subcategory == UNKNOWN_SUBCATEGORY and title is not None:
                    if len(tops_unknown_title_samples) < UNKNOWN_TITLE_SAMPLE_LIMIT:
                        tops_unknown_title_samples.append(title)

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
        ranking_group_counts=ranking_group_counts,
        tops_subcategory_counts=tops_subcategory_counts,
        apparel_counts=apparel_counts,
        source_counts=source_counts,
        dropped_rows=dropped_rows,
        input_files=input_files,
        output_path=output_path,
        unknown_title_samples=unknown_title_samples,
        tops_unknown_title_samples=tops_unknown_title_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
