import os
import sys
import gradio as gr
import yaml
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging

from csi_models.bytecover_utils import compute_similarity_bytecover, load_bytecover_model
from csi_models.coverhunter_utils import compute_similarity_coverhunter, load_coverhunter_model
from csi_models.lyricover_utils import compute_similarity_lyricover, load_lyricover_model
from feature_extraction.feature_extraction import compute_similarity
import tqdm
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("cover_song_evaluation.log")
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

with open("configs/paths.yaml", "r") as f:
    config = yaml.safe_load(f)

COVERS80_DATA_DIR = config["covers80_data_dir"]
COVERS80BUT10_DATA_DIR = config["covers80but10_data_dir"]

def gather_covers80_dataset_files(dataset_path):
    """
    Return a list of (audio_file_path, label) for all folders in covers80 or covers80but10.
    """
    all_audio_files = []
    song_labels = []

    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue

        audio_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path) if f.endswith(".mp3")
        ]
        if len(audio_files) < 2:
            logger.warning(f"Skipping folder {folder} with less than 2 files.")
            continue

        all_audio_files.extend(audio_files)
        song_labels.extend([folder] * len(audio_files))

    return list(zip(all_audio_files, song_labels))

from tqdm import tqdm
def compute_rankings_per_song(files_and_labels, similarity_function, progress=gr.Progress()):
    """
    For each song in 'files_and_labels', create a separate ranking 
    based on similarity to other songs.
    
    Returns a list of dictionaries:
    [
      {
        "query_path": ...,
        "query_label": ...,
        "ranking": [
          { "candidate_path": ..., "similarity": float, "ground_truth": bool },
          ...
        ]
      },
      ...
    ]
    """
    rankings_per_query = []
    total_comparisons = len(files_and_labels)
    progress(0, desc="Initializing pairwise comparisons")
    
    for i, (query_path, query_label) in enumerate(tqdm(files_and_labels)):
        # Build a list of comparisons to all other songs
        comparisons = []
        for j, (cand_path, cand_label) in enumerate(files_and_labels):
            if i == j:
                continue  # Skip the same song

            sim = similarity_function(query_path, cand_path)
            gt = (query_label == cand_label)  # True if it is indeed a cover
            comparisons.append({
                "candidate_path": cand_path,
                "similarity": sim,
                "ground_truth": gt
            })

        # Sort descending by similarity
        comparisons.sort(key=lambda x: x["similarity"], reverse=True)

        # Save the ranking for query_path
        rankings_per_query.append({
            "query_path": query_path,
            "query_label": query_label,
            "ranking": comparisons
        })
        progress((i + 1) / total_comparisons, desc="Evaluating pairs")

    return rankings_per_query

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


def evaluate_on_covers80(
    model_name,
    covers80but10=False,
    k=10
):
    """
    Loads the covers80 (or covers80but10) dataset, computes rankings per song,
    and calculates mAP, mP@k, mMR1.
    """

    # 1. Select dataset path
    if covers80but10:
        dataset_path = COVERS80BUT10_DATA_DIR
    else:
        dataset_path = COVERS80_DATA_DIR

    # 2. Prepare similarity_function depending on the model
    if model_name == "ByteCover":
        model = load_bytecover_model()
        def similarity_function(a, b):
            return compute_similarity_bytecover(a, b, model)
    elif model_name == "CoverHunter":
        model = load_coverhunter_model()
        def similarity_function(a, b):
            return compute_similarity_coverhunter(a, b, model)
    elif model_name == "Lyricover":
        model = load_lyricover_model()
        def similarity_function(a, b):
            return compute_similarity_lyricover(a, b, model)
    elif model_name in ["MFCC", "Spectral Centroid"]:
        def similarity_function(a, b):
            return compute_similarity(a, b, model_name)
        model = None
    else:
        raise ValueError("Unsupported model. Choose ByteCover, CoverHunter, Lyricover, MFCC or Spectral Centroid.")

    # 3. Load the list of (file, label)
    files_and_labels = gather_covers80_dataset_files(dataset_path)

    # 4. Ranking for each song
    rankings_per_query = compute_rankings_per_song(files_and_labels, similarity_function)

    # 5. Compute aggregate metrics
    metrics = compute_mean_metrics_for_rankings(rankings_per_query, k=k)
    
    # 6. Return results
    return {
        "Model": model_name,
        "Dataset": "covers80but10" if covers80but10 else "covers80",
        "Mean Average Precision (mAP)": metrics["mAP"],
        f"Precision at {k} (mP@{k})": metrics["mP@k"],
        "Mean Rank of First Correct Cover (mMR1)": metrics["mMR1"]
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate cover song detection models on the covers80 dataset.")
    
    # Add argument for model selection
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["ByteCover", "CoverHunter", "Lyricover", "MFCC", "Spectral Centroid"],
        help="The name of the model to evaluate (ByteCover, CoverHunter, Lyricover, MFCC, Spectral Centroid)."
    )

    args = parser.parse_args()

    result = evaluate_on_covers80(
        model_name=args.model
    )
    print("Evaluation Results:")
    for key, value in result.items():
        print(f"{key}: {value}")