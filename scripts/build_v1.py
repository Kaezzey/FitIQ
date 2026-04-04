from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    ("Build canonical reviews", ["data/build_reviews_table.py"]),
    ("Build product taxonomy", ["data/build_product_taxonomy.py"]),
    ("Build product features", ["data/build_product_features.py"]),
    ("Build ranking evaluation dataset", ["data/build_ranking_eval_dataset.py"]),
    ("Run baseline benchmarks", ["evaluation/run_baseline_benchmarks.py"]),
    ("Train CatBoost ranker", ["models/train_catboost_ranker.py"]),
    ("Tune CatBoost ranker", ["models/tune_catboost_ranker.py"]),
    ("Build query-group hybrid champion", ["models/train_query_group_hybrid_ranker.py"]),
    ("Score full-corpus product rankings", ["models/score_product_rankings.py"]),
]

EXPECTED_OUTPUTS = [
    PROJECT_ROOT / "data/processed/product_rankings_v1.parquet",
    PROJECT_ROOT / "data/processed/product_rankings_v1_summary.json",
    PROJECT_ROOT / "data/processed/fitiq_v1_manifest.json",
]


def run_step(index: int, total: int, label: str, command: list[str]) -> None:
    print()
    print(f"[{index}/{total}] {label}")
    print(f"  command: {sys.executable} {' '.join(command)}")
    completed = subprocess.run([sys.executable, *command], cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Step failed with exit code {completed.returncode}: {label}")


def main() -> int:
    print("FitIQ v1 rebuild starting...")
    for index, (label, command) in enumerate(STEPS, start=1):
        run_step(index=index, total=len(STEPS), label=label, command=command)

    missing_outputs = [path for path in EXPECTED_OUTPUTS if not path.exists()]
    if missing_outputs:
        raise RuntimeError(
            "FitIQ v1 rebuild completed but expected outputs were missing:\n"
            + "\n".join(f"  - {path}" for path in missing_outputs)
        )

    print()
    print("FitIQ v1 rebuild complete.")
    for path in EXPECTED_OUTPUTS:
        print(f"  verified: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
