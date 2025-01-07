import os
import gradio as gr
import yaml

from csi_models.bytecover_utils import compute_similarity_bytecover, load_bytecover_model
from csi_models.coverhunter_utils import compute_similarity_coverhunter, load_coverhunter_model
from csi_models.lyricover_utils import compute_similarity_lyricover, load_lyricover_model
from feature_extraction.feature_extraction import compute_similarity
from evaluation.metrics import compute_mean_metrics_for_rankings
from tqdm import tqdm
from utils.logging_config import logger

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
