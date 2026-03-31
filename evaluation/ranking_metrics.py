from __future__ import annotations

import math


def _validate_columns(frame, required_columns: set[str]) -> None:
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"Frame is missing required columns: {', '.join(missing)}")


def dcg_at_k(relevances: list[float], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")
    total = 0.0
    for index, relevance in enumerate(relevances[:k], start=1):
        gain = (2.0**float(relevance)) - 1.0
        total += gain / math.log2(index + 1.0)
    return total


def ndcg_at_k(y_true: list[float], y_score: list[float], k: int) -> float:
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length.")
    if not y_true:
        return 0.0

    ranked_pairs = sorted(
        zip(y_true, y_score, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    ranked_relevances = [float(relevance) for relevance, _ in ranked_pairs]
    ideal_relevances = sorted((float(value) for value in y_true), reverse=True)

    ideal_dcg = dcg_at_k(ideal_relevances, k)
    if ideal_dcg <= 0.0:
        return 0.0
    return dcg_at_k(ranked_relevances, k) / ideal_dcg


def _average_ranks(values: list[float]) -> list[float]:
    indexed_values = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed_values):
        next_index = index + 1
        while (
            next_index < len(indexed_values)
            and indexed_values[next_index][1] == indexed_values[index][1]
        ):
            next_index += 1
        average_rank = ((index + 1) + next_index) / 2.0
        for original_index, _ in indexed_values[index:next_index]:
            ranks[original_index] = average_rank
        index = next_index
    return ranks


def _pearson_correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length.")
    if len(xs) < 2:
        return 0.0

    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    x_variance = sum((x - x_mean) ** 2 for x in xs)
    y_variance = sum((y - y_mean) ** 2 for y in ys)
    denominator = math.sqrt(x_variance * y_variance)
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def spearman_rank_correlation(y_true: list[float], y_score: list[float]) -> float:
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length.")
    if len(y_true) < 2:
        return 0.0
    truth_ranks = _average_ranks([float(value) for value in y_true])
    score_ranks = _average_ranks([float(value) for value in y_score])
    return _pearson_correlation(truth_ranks, score_ranks)


def grouped_ndcg_at_k(frame, group_col: str, truth_col: str, score_col: str, k: int) -> float:
    _validate_columns(frame, {group_col, truth_col, score_col})
    scores: list[float] = []
    for _, group_frame in frame.groupby(group_col, sort=False):
        scores.append(
            ndcg_at_k(
                group_frame[truth_col].astype(float).tolist(),
                group_frame[score_col].astype(float).tolist(),
                k=k,
            )
        )
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def grouped_spearman(frame, group_col: str, truth_col: str, score_col: str) -> float:
    _validate_columns(frame, {group_col, truth_col, score_col})
    scores: list[float] = []
    for _, group_frame in frame.groupby(group_col, sort=False):
        scores.append(
            spearman_rank_correlation(
                group_frame[truth_col].astype(float).tolist(),
                group_frame[score_col].astype(float).tolist(),
            )
        )
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
