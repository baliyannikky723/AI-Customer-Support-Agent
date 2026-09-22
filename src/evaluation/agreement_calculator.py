"""
Inter-Rater and Judge-Human Agreement Calculator.

Computes:
- Exact Agreement Rate (%)
- Within-1 Agreement Rate (%)
- Mean Absolute Difference (MAD)
- Weighted Cohen's Kappa (Quadratic & Linear)
- Unweighted Cohen's Kappa for Binary Targets
- Pearson and Spearman Correlation Coefficients
"""

from typing import List, Dict, Tuple, Optional
import numpy as np

from src.evaluation.judge.judge_schema import (
    JudgeScore,
    HumanAnnotation,
    DimensionAgreement,
    BinaryAgreement,
    AgreementReport,
)


def compute_weighted_cohen_kappa(
    rater1: List[int],
    rater2: List[int],
    min_rating: int = 1,
    max_rating: int = 5,
    weight_type: str = "quadratic",
) -> float:
    """Compute weighted Cohen's Kappa for ordinal ratings (1-5).
    
    Args:
        rater1: List of integer scores from rater 1.
        rater2: List of integer scores from rater 2.
        min_rating: Lowest possible score (default 1).
        max_rating: Highest possible score (default 5).
        weight_type: 'linear' or 'quadratic' weighting.
        
    Returns:
        Weighted Cohen's Kappa statistic in range [-1.0, 1.0].
    """
    n = len(rater1)
    if n == 0 or len(rater2) != n:
        return 0.0

    k = max_rating - min_rating + 1
    # Build observed confusion matrix
    observed_matrix = np.zeros((k, k), dtype=float)
    for r1, r2 in zip(rater1, rater2):
        i = np.clip(r1 - min_rating, 0, k - 1)
        j = np.clip(r2 - min_rating, 0, k - 1)
        observed_matrix[i, j] += 1.0

    # Build weight matrix
    weight_matrix = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            if weight_type == "linear":
                weight_matrix[i, j] = 1.0 - (abs(i - j) / (k - 1))
            else:  # quadratic
                weight_matrix[i, j] = 1.0 - (((i - j) ** 2) / ((k - 1) ** 2))

    # Compute expected matrix
    r1_dist = np.sum(observed_matrix, axis=1) / n
    r2_dist = np.sum(observed_matrix, axis=0) / n
    expected_matrix = np.outer(r1_dist, r2_dist) * n

    po = np.sum(weight_matrix * observed_matrix) / n
    pe = np.sum(weight_matrix * expected_matrix) / n

    if pe == 1.0:
        return 1.0

    kappa = (po - pe) / (1.0 - pe)
    return float(np.clip(round(kappa, 4), -1.0, 1.0))


def compute_binary_cohen_kappa(
    rater1: List[bool],
    rater2: List[bool],
) -> float:
    """Compute unweighted Cohen's Kappa for binary categories."""
    n = len(rater1)
    if n == 0 or len(rater2) != n:
        return 0.0

    # 2x2 confusion matrix: [ [TN, FP], [FN, TP] ]
    cm = np.zeros((2, 2), dtype=float)
    for r1, r2 in zip(rater1, rater2):
        i = 1 if r1 else 0
        j = 1 if r2 else 0
        cm[i, j] += 1.0

    po = (cm[0, 0] + cm[1, 1]) / n
    r1_pos = (cm[1, 0] + cm[1, 1]) / n
    r1_neg = 1.0 - r1_pos
    r2_pos = (cm[0, 1] + cm[1, 1]) / n
    r2_neg = 1.0 - r2_pos

    pe = (r1_pos * r2_pos) + (r1_neg * r2_neg)

    if pe == 1.0:
        return 1.0

    kappa = (po - pe) / (1.0 - pe)
    return float(np.clip(round(kappa, 4), -1.0, 1.0))


def compute_dimension_agreement(
    dim_name: str,
    rater1_scores: List[int],
    rater2_scores: List[int],
) -> DimensionAgreement:
    """Compute agreement metrics for a single 1-5 rubric dimension."""
    n = len(rater1_scores)
    if n == 0:
        return DimensionAgreement(
            dimension_name=dim_name,
            exact_agreement_pct=0.0,
            within_one_pct=0.0,
            mean_absolute_difference=0.0,
            weighted_cohen_kappa=0.0,
            rater1_mean=0.0,
            rater2_mean=0.0,
        )

    r1_arr = np.array(rater1_scores, dtype=float)
    r2_arr = np.array(rater2_scores, dtype=float)

    exact = float(np.mean(r1_arr == r2_arr) * 100.0)
    within_one = float(np.mean(np.abs(r1_arr - r2_arr) <= 1.0) * 100.0)
    mad = float(np.mean(np.abs(r1_arr - r2_arr)))
    kappa = compute_weighted_cohen_kappa(rater1_scores, rater2_scores)

    # Pearson correlation
    pearson = None
    if np.std(r1_arr) > 1e-6 and np.std(r2_arr) > 1e-6:
        pearson = float(np.corrcoef(r1_arr, r2_arr)[0, 1])

    return DimensionAgreement(
        dimension_name=dim_name,
        exact_agreement_pct=round(exact, 2),
        within_one_pct=round(within_one, 2),
        mean_absolute_difference=round(mad, 4),
        weighted_cohen_kappa=round(kappa, 4),
        pearson_correlation=round(pearson, 4) if pearson is not None else None,
        rater1_mean=round(float(np.mean(r1_arr)), 2),
        rater2_mean=round(float(np.mean(r2_arr)), 2),
    )


def compute_agreement_report(
    rater1_name: str,
    rater2_name: str,
    rater1_scores: List[JudgeScore],
    rater2_scores: List[HumanAnnotation],
) -> AgreementReport:
    """Compute comprehensive agreement report between two sets of ratings."""
    n = min(len(rater1_scores), len(rater2_scores))
    if n == 0:
        return AgreementReport(
            num_evaluated_pairs=0,
            rater1_name=rater1_name,
            rater2_name=rater2_name,
            status="NO_DATA",
            notes=["No paired evaluation examples available."],
        )

    r1_subset = rater1_scores[:n]
    r2_subset = rater2_scores[:n]

    dim_agreements: Dict[str, DimensionAgreement] = {}
    dimensions = ["correctness", "groundedness", "helpfulness", "safety", "tone", "overall_quality"]

    kappas = []
    for dim in dimensions:
        r1_vals = [getattr(s, dim) for s in r1_subset]
        r2_vals = [getattr(s, dim) for s in r2_subset]
        da = compute_dimension_agreement(dim, r1_vals, r2_vals)
        dim_agreements[dim] = da
        kappas.append(da.weighted_cohen_kappa)

    # Binary metrics
    binary_agreements: Dict[str, BinaryAgreement] = {}
    for bmetric in ["acceptable", "critical_failure"]:
        r1_bvals = [getattr(s, bmetric) for s in r1_subset]
        r2_bvals = [getattr(s, bmetric) for s in r2_subset]
        b_kappa = compute_binary_cohen_kappa(r1_bvals, r2_bvals)
        exact_b = float(np.mean(np.array(r1_bvals) == np.array(r2_bvals)) * 100.0)
        binary_agreements[bmetric] = BinaryAgreement(
            metric_name=bmetric,
            exact_agreement_pct=round(exact_b, 2),
            cohen_kappa=round(b_kappa, 4),
            rater1_positive_count=sum(1 for v in r1_bvals if v),
            rater2_positive_count=sum(1 for v in r2_bvals if v),
        )

    mean_kappa = float(np.mean(kappas))

    return AgreementReport(
        num_evaluated_pairs=n,
        rater1_name=rater1_name,
        rater2_name=rater2_name,
        dimension_agreements=dim_agreements,
        binary_agreements=binary_agreements,
        mean_overall_kappa=round(mean_kappa, 4),
        status="COMPLETED",
        notes=[f"Evaluated {n} paired annotations across 6 ordinal dimensions and 2 binary indicators."],
    )
