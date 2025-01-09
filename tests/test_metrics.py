import pytest
from evaluation.metrics import compute_metrics_for_ranking, compute_mean_metrics_for_rankings
from typing import List, Dict

# Tests for compute_metrics_for_ranking
def test_compute_metrics_for_ranking_no_relevant():
    ranking = [{"ground_truth": False} for _ in range(5)]
    result = compute_metrics_for_ranking(ranking)
    assert result["AP"] == 0.0
    assert result["P@k"] == 0.0
    assert result["R1"] == len(ranking) + 1

def test_compute_metrics_for_ranking_all_relevant():
    ranking = [{"ground_truth": True} for _ in range(5)]
    result = compute_metrics_for_ranking(ranking)
    assert result["AP"] == 1.0
    assert result["P@k"] == 1.0
    assert result["R1"] == 1

def test_compute_metrics_for_ranking_mixed():
    ranking = [
        {"ground_truth": False},
        {"ground_truth": True},
        {"ground_truth": False},
        {"ground_truth": True},
        {"ground_truth": True},
        {"ground_truth": False}
    ]
    result = compute_metrics_for_ranking(ranking, k=3)
    assert abs(result["AP"] - 0.5333) < 0.0001
    assert abs(result["P@k"] - 0.3333) < 0.0001
    assert result["R1"] == 2

# Tests for compute_mean_metrics_for_rankings
def test_compute_mean_metrics_for_rankings_empty():
    rankings_per_query = []
    result = compute_mean_metrics_for_rankings(rankings_per_query, k=10)
    assert result["mAP"] == 0.0
    assert result["mP@k"] == 0.0
    assert result["mMR1"] == 0.0

def test_compute_mean_metrics_for_rankings_single():
    rankings_per_query = [
        {
            "ranking": [
                {"ground_truth": True},
                {"ground_truth": False},
                {"ground_truth": True},
            ]
        }
    ]
    single_result = compute_metrics_for_ranking(rankings_per_query[0]["ranking"], k=2)
    mean_result = compute_mean_metrics_for_rankings(rankings_per_query, k=2)
    assert mean_result["mAP"] == single_result["AP"]
    assert mean_result["mP@k"] == single_result["P@k"]
    assert mean_result["mMR1"] == single_result["R1"]

def test_compute_mean_metrics_for_rankings_multiple():
    rankings_per_query = [
        {
            "ranking": [
                {"ground_truth": True},
                {"ground_truth": False},
                {"ground_truth": False},
                {"ground_truth": True},
            ]
        },
        {
            "ranking": [
                {"ground_truth": False},
                {"ground_truth": False},
            ]
        },
        {
            "ranking": [
                {"ground_truth": True},
                {"ground_truth": True},
            ]
        }
    ]
    individual_metrics = [
        compute_metrics_for_ranking(rpq["ranking"], k=2) for rpq in rankings_per_query
    ]
    mean_AP_expected = sum(m["AP"] for m in individual_metrics) / len(individual_metrics)
    mean_Pk_expected = sum(m["P@k"] for m in individual_metrics) / len(individual_metrics)
    mean_R1_expected = sum(m["R1"] for m in individual_metrics) / len(individual_metrics)
    mean_result = compute_mean_metrics_for_rankings(rankings_per_query, k=2)
    assert abs(mean_result["mAP"] - mean_AP_expected) < 1e-6
    assert abs(mean_result["mP@k"] - mean_Pk_expected) < 1e-6
    assert abs(mean_result["mMR1"] - mean_R1_expected) < 1e-6
