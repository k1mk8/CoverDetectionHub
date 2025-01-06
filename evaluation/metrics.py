def calculate_precision_at_k(results, k=10):
    """Calculate Precision@k given a list of comparison results."""
    top_k = results[:k]
    if len(top_k) == 0:
        return 0.0
    correct = sum(1 for r in top_k if r["is_cover"] == r["ground_truth"])
    return correct / len(top_k)

def calculate_mean_rank_of_first_correct_cover(results):
    """Calculate Mean Rank of First Correct Cover (MR1)."""
    first_correct_ranks = [
        i + 1 for i, r in enumerate(results)
        if r["is_cover"] and r["ground_truth"]
    ]
    if first_correct_ranks:
        return sum(first_correct_ranks) / len(first_correct_ranks)
    return float('inf')

def calculate_mean_average_precision(results):
    """Calculate mAP given a list of comparison results."""
    total_relevant = sum(1 for r in results if r["ground_truth"])
    if total_relevant == 0:
        return 0.0

    predicted_correct_count = 0
    precision_sum = 0.0
    for i, r in enumerate(results):
        if r["is_cover"] and r["ground_truth"]:
            predicted_correct_count += 1
            precision_at_i = predicted_correct_count / (i + 1)
            precision_sum += precision_at_i

    return precision_sum / total_relevant