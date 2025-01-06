import pytest
from evaluation.metrics import (
    calculate_precision_at_k,
    calculate_mean_rank_of_first_correct_cover,
    calculate_mean_average_precision
)

# Sample data for testing
@pytest.fixture
def sample_results():
    return [
        {"is_cover": True,  "ground_truth": True},   # (1) Correctly predicted as cover
        {"is_cover": False, "ground_truth": True},   # (2) Missed a true cover
        {"is_cover": True,  "ground_truth": False},  # (3) False positive
        {"is_cover": True,  "ground_truth": True},   # (4) Correctly predicted as cover
        {"is_cover": False, "ground_truth": False},  # (5) Correctly predicted as not a cover
    ]



def test_calculate_precision_at_k(sample_results):
    """Test calculate_precision_at_k."""
    # Top-3 precision
    precision_at_3 = calculate_precision_at_k(sample_results, k=3)
    assert precision_at_3 == pytest.approx(1 / 3), "Precision@3 should be 1/3."

    # Top-5 precision
    precision_at_5 = calculate_precision_at_k(sample_results, k=5)
    assert precision_at_5 == pytest.approx(3 / 5)  # i.e., 0.4

    # No results (edge case)
    precision_at_0 = calculate_precision_at_k([], k=3)
    assert precision_at_0 == 0.0, "Precision@k with no results should be 0."


def test_calculate_mean_rank_of_first_correct_cover(sample_results):
    """Test calculate_mean_rank_of_first_correct_cover."""
    mr1 = calculate_mean_rank_of_first_correct_cover(sample_results)
    assert mr1 == pytest.approx((1 + 4) / 2), "Mean rank of first correct cover should be (1 + 4)/2."

    # No correct covers
    no_correct_results = [{"is_cover": False, "ground_truth": True}] * 5
    mr1 = calculate_mean_rank_of_first_correct_cover(no_correct_results)
    assert mr1 == float('inf'), "MR1 should be infinity if there are no correct covers."

    # Edge case: single correct cover
    single_correct_result = [{"is_cover": True, "ground_truth": True}]
    mr1 = calculate_mean_rank_of_first_correct_cover(single_correct_result)
    assert mr1 == 1.0, "MR1 for a single correct cover should be 1."


def test_calculate_mean_average_precision(sample_results):
    """Test calculate_mean_average_precision."""
    # Total relevant = 2, Correct at ranks: 1 (P@1=1.0), 4 (P@4=2/4)
    # mAP = (1.0 + 0.5) / 2 = 0.75
    mAP = calculate_mean_average_precision(sample_results)
    assert mAP == pytest.approx(0.5), "mAP should be 0.5 for the given results."

    # No relevant results
    no_relevant_results = [{"is_cover": False, "ground_truth": False}] * 5
    mAP = calculate_mean_average_precision(no_relevant_results)
    assert mAP == 0.0, "mAP should be 0 if there are no relevant results."

    # Edge case: perfect precision
    perfect_results = [{"is_cover": True, "ground_truth": True}] * 5
    mAP = calculate_mean_average_precision(perfect_results)
    assert mAP == 1.0, "mAP should be 1.0 if all predictions are correct."
