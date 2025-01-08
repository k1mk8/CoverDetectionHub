
def compute_metrics_for_ranking(ranking, k=10):
    """
    Computes AP, P@k (default k=10), and the position of the first correct cover (R1)
    for a single ranking sorted in descending similarity order.
    """
    # ---- 1. Number of actual covers (ground_truth == True)
    total_relevant = sum(1 for item in ranking if item["ground_truth"])
    N = len(ranking)

    # ---- 2. AP (Average Precision)
    if total_relevant == 0:
        AP = 0.0
    else:
        precision_sum = 0.0
        predicted_correct_count = 0
        # Iterate through ranking elements
        for i, item in enumerate(ranking):
            if item["ground_truth"]:
                predicted_correct_count += 1
                precision_at_i = predicted_correct_count / (i + 1)  # i+1 because i starts at 0
                precision_sum += precision_at_i
        AP = precision_sum / total_relevant

    # ---- 3. P@k (Precision at K)
    # Take the top-k elements from the ranking
    top_k = ranking[:k] if k <= N else ranking
    relevant_in_top_k = sum(1 for item in top_k if item["ground_truth"])
    Pk = relevant_in_top_k / float(k) if k <= N else relevant_in_top_k / float(N)

    # ---- 4. R1 (Rank of the first correct cover)
    # Find the first position in the ranking with ground_truth=True
    first_correct_cover_index = None
    for i, item in enumerate(ranking):
        if item["ground_truth"]:
            first_correct_cover_index = i
            break
    # If no matches, assign rank = N+1 (or another large value)
    if first_correct_cover_index is None:
        R1 = N + 1
    else:
        # Rank is i+1 (to get "1-based rank", not 0-based index)
        R1 = first_correct_cover_index + 1

    return {
        "AP": AP,
        "P@k": Pk,
        "R1": R1
    }


def compute_mean_metrics_for_rankings(rankings_per_query, k=10):
    """
    For a list of rankings (each related to a different song), compute:
      - mAP (Mean Average Precision)
      - mP@k (Mean Precision @k)
      - mMR1 (Mean rank of the first correct match)
    and return them as a dictionary.
    """
    AP_values = []
    Pk_values = []
    R1_values = []

    for entry in rankings_per_query:
        ranking = entry["ranking"]
        metrics = compute_metrics_for_ranking(ranking, k=k)
        AP_values.append(metrics["AP"])
        Pk_values.append(metrics["P@k"])
        R1_values.append(metrics["R1"])

    # Avoid ZeroDivisionError if lists are empty for any reason
    if len(AP_values) == 0:
        return {
            "mAP": 0.0,
            "mP@k": 0.0,
            "mMR1": 0.0
        }

    mean_AP = sum(AP_values) / len(AP_values)
    mean_Pk = sum(Pk_values) / len(Pk_values)
    mean_R1 = sum(R1_values) / len(R1_values)

    return {
        "mAP": mean_AP,
        "mP@k": mean_Pk,
        "mMR1": mean_R1
    }
