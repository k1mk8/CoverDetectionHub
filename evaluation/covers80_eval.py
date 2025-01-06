

import os
import gradio as gr
import yaml

from csi_models.bytecover_utils import compute_similarity_bytecover, load_bytecover_model
from csi_models.coverhunter_utils import compute_similarity_coverhunter, load_coverhunter_model
from csi_models.lyricover_utils import compute_similarity_lyricover, load_lyricover_model
from feature_extraction.feature_extraction import compute_similarity
from evaluation.metrics import calculate_mean_average_precision, calculate_mean_rank_of_first_correct_cover, calculate_precision_at_k
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

def compute_covers80_results(files_and_labels, similarity_function, threshold, progress=gr.Progress()):
    """
    Given a list of (file_path, label) tuples, compute all pairwise similarities
    using similarity_function. Return a list of result dictionaries.
    """
    results = []
    num_files = len(files_and_labels)
    file_pairs = [
        (files_and_labels[i], files_and_labels[j])
        for i in range(num_files) for j in range(i + 1, num_files)
    ]

    total_comparisons = len(file_pairs)
    progress(0, desc="Initializing pairwise comparisons")

    for idx, pair in enumerate(file_pairs):
        (song_a_path, label_a), (song_b_path, label_b) = pair
        similarity = similarity_function(song_a_path, song_b_path)

        is_cover = (similarity >= threshold)
        ground_truth = (label_a == label_b)  # same folder => same song => cover

        # Log comparison details
        logger.info(f"Comparison {idx + 1}/{total_comparisons}")
        logger.info(f"  Song A: {song_a_path}")
        logger.info(f"  Song B: {song_b_path}")
        logger.info(f"  Similarity: {similarity:.4f}")
        logger.info(f"  Predicted Cover: {'Yes' if is_cover else 'No'}")
        logger.info(f"  Ground Truth Cover: {'Yes' if ground_truth else 'No'}")

        results.append({
            "song_pair": (song_a_path, song_b_path),
            "similarity": similarity,
            "is_cover": is_cover,
            "ground_truth": ground_truth
        })

        progress((idx + 1) / total_comparisons, desc="Evaluating pairs")

    # Sort descending by similarity
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results


def evaluate_on_covers80(model_name, threshold=0.99, covers80but10=False, progress=gr.Progress()):
    """
    Evaluate any of the five models (ByteCover, CoverHunter, Lyricover, MFCC, Spectral Centroid)
    on the covers80 or covers80but10 dataset.
    """
    if covers80but10:
        dataset_path = COVERS80BUT10_DATA_DIR
    else:
        dataset_path = COVERS80_DATA_DIR

    # Load model or define similarity function
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
        # We pass model_name in closure
        def similarity_function(a, b):
            return compute_similarity(a, b, model_name)
        model = None
    else:
        raise ValueError("Unsupported model. Choose ByteCover, CoverHunter, Lyricover, MFCC or Spectral Centroid.")

    # Gather data and compute results
    files_and_labels = gather_covers80_dataset_files(dataset_path)
    results = compute_covers80_results(files_and_labels, similarity_function, threshold, progress)

    # Calculate metrics
    p_at_10 = calculate_precision_at_k(results, k=10)
    mr1 = calculate_mean_rank_of_first_correct_cover(results)
    mAP = calculate_mean_average_precision(results)

    # Additional stats
    total_relevant = sum(1 for r in results if r["ground_truth"])
    predicted_correct_count = sum(1 for r in results if r["is_cover"] and r["ground_truth"])

    summary_metrics = {
        "Mean Average Precision (mAP)": mAP,
        "Precision at 10 (P@10)": p_at_10,
        "Mean Rank of First Correct Cover (MR1)": mr1,
        "Total Covers Predicted Correctly": predicted_correct_count,
        "Total Pairs (Ground Truth Covers)": total_relevant,
        "Threshold Used": threshold
    }

    return summary_metrics
