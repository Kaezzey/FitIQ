from __future__ import annotations

import pandas as pd

import ranking_metrics


def build_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"subcategory": "pants", "asin": "A1", "target_score": 0.20},
            {"subcategory": "pants", "asin": "A2", "target_score": 0.95},
            {"subcategory": "pants", "asin": "A3", "target_score": 0.40},
            {"subcategory": "pants", "asin": "A4", "target_score": 0.75},
            {"subcategory": "jacket", "asin": "B1", "target_score": 0.85},
            {"subcategory": "jacket", "asin": "B2", "target_score": 0.15},
            {"subcategory": "jacket", "asin": "B3", "target_score": 0.65},
            {"subcategory": "jacket", "asin": "B4", "target_score": 0.35},
        ]
    )


def main() -> int:
    fixture = build_fixture()

    oracle = fixture.assign(score=fixture["target_score"])
    constant = fixture.assign(score=0.5)
    reversed_rank = fixture.assign(
        score=[0.95, 0.20, 0.75, 0.40, 0.15, 0.85, 0.35, 0.65]
    )
    shuffled = fixture.assign(score=[0.40, 0.95, 0.20, 0.75, 0.65, 0.15, 0.85, 0.35])

    oracle_ndcg10 = ranking_metrics.grouped_ndcg_at_k(
        oracle,
        group_col="subcategory",
        truth_col="target_score",
        score_col="score",
        k=10,
    )
    constant_ndcg10 = ranking_metrics.grouped_ndcg_at_k(
        constant,
        group_col="subcategory",
        truth_col="target_score",
        score_col="score",
        k=10,
    )
    reversed_ndcg10 = ranking_metrics.grouped_ndcg_at_k(
        reversed_rank,
        group_col="subcategory",
        truth_col="target_score",
        score_col="score",
        k=10,
    )
    shuffled_ndcg10 = ranking_metrics.grouped_ndcg_at_k(
        shuffled,
        group_col="subcategory",
        truth_col="target_score",
        score_col="score",
        k=10,
    )

    oracle_spearman = ranking_metrics.grouped_spearman(
        oracle,
        group_col="subcategory",
        truth_col="target_score",
        score_col="score",
    )
    constant_spearman = ranking_metrics.grouped_spearman(
        constant,
        group_col="subcategory",
        truth_col="target_score",
        score_col="score",
    )
    reversed_spearman = ranking_metrics.grouped_spearman(
        reversed_rank,
        group_col="subcategory",
        truth_col="target_score",
        score_col="score",
    )
    shuffled_spearman = ranking_metrics.grouped_spearman(
        shuffled,
        group_col="subcategory",
        truth_col="target_score",
        score_col="score",
    )

    assert oracle_ndcg10 > constant_ndcg10
    assert oracle_ndcg10 > reversed_ndcg10
    assert oracle_ndcg10 > shuffled_ndcg10
    assert oracle_spearman > constant_spearman
    assert oracle_spearman > reversed_spearman
    assert oracle_spearman > shuffled_spearman

    print("Ranking metric smoke tests passed.")
    print(f"  oracle_ndcg10: {oracle_ndcg10:.4f}")
    print(f"  constant_ndcg10: {constant_ndcg10:.4f}")
    print(f"  reversed_ndcg10: {reversed_ndcg10:.4f}")
    print(f"  shuffled_ndcg10: {shuffled_ndcg10:.4f}")
    print(f"  oracle_spearman: {oracle_spearman:.4f}")
    print(f"  constant_spearman: {constant_spearman:.4f}")
    print(f"  reversed_spearman: {reversed_spearman:.4f}")
    print(f"  shuffled_spearman: {shuffled_spearman:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
