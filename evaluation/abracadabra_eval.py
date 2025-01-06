


import os
import pandas as pd
import gradio as gr
import yaml
from csi_models.bytecover_utils import compute_similarity_bytecover, load_bytecover_model
from csi_models.coverhunter_utils import compute_similarity_coverhunter, load_coverhunter_model
from evaluation.metrics import calculate_mean_average_precision, calculate_mean_rank_of_first_correct_cover, calculate_precision_at_k
from feature_extraction.feature_extraction import compute_similarity
from utils.logging_config import logger


with open("configs/paths.yaml", "r") as f:
    config = yaml.safe_load(f)

INJECTED_ABRACADABRA_DIR = config["injected_abracadabra_dir"]
INJECTED_ABRACADABRA_DATA_DIR = config["injected_abracadabra_data_dir"]
GROUND_TRUTH_FILE = config["injected_abracadabra_ground_truth_file"]
REFERENCE_SONG = config["injected_abracadabra_reference_song"]


def gather_injected_abracadabra_files(dataset_path, ground_truth_file):
    """
    Return (file_path, is_injected) pairs by reading from the injection_list CSV.
    """
    injection_data = pd.read_csv(ground_truth_file)
    # Normalize the path by replacing backslashes
    injection_dict = dict(
        zip(
            injection_data['File'].str.replace("\\", "/"),
            injection_data['Injected'].map(lambda x: x == 'Yes')
        )
    )

    all_audio_files = [
        os.path.join(dataset_path, f)
        for f in os.listdir(dataset_path) if f.endswith(".wav")
    ]

    # Build a list of (path, ground_truth_is_injected)
    files_with_ground_truth = []
    for path in all_audio_files:
        
        rel_path = os.path.relpath(path, start=INJECTED_ABRACADABRA_DIR).replace("\\", "/")

        # Look up in injection_dict
        is_injected = injection_dict.get(rel_path, False)
        files_with_ground_truth.append((path, is_injected))

    return files_with_ground_truth


def compute_injected_abracadabra_results(files_with_ground_truth, reference_song, similarity_function, threshold, progress=gr.Progress()):
    """
    Compare each file in 'files_with_ground_truth' to the reference song.
    """
    results = []
    total_files = len(files_with_ground_truth)
    progress(0, desc="Initializing comparisons")

    for idx, (song_path, is_injected) in enumerate(files_with_ground_truth):
        similarity = similarity_function(song_path, reference_song)

        predicted_cover = (similarity >= threshold)

        logger.info(f"Comparison {idx + 1}/{total_files}")
        logger.info(f"  Song: {song_path}")
        logger.info(f"  Similarity: {similarity:.4f}")
        logger.info(f"  Predicted Cover: {'Yes' if predicted_cover else 'No'}")
        logger.info(f"  Ground Truth Cover: {'Yes' if is_injected else 'No'}")

        results.append({
            "song": song_path,
            "similarity": similarity,
            "is_cover": predicted_cover,
            "ground_truth": is_injected
        })
        progress((idx + 1) / total_files, desc="Evaluating files")

    # Sort descending by similarity
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results

def evaluate_on_injected_abracadabra(model_name, threshold=0.99, progress=gr.Progress()):

    dataset_path = INJECTED_ABRACADABRA_DATA_DIR
    ground_truth_file = GROUND_TRUTH_FILE
    reference_song = REFERENCE_SONG

    # Load model or define similarity function
    if model_name == "ByteCover":
        model = load_bytecover_model()
        def similarity_function(a, b):
            return compute_similarity_bytecover(a, b, model)
    elif model_name == "CoverHunter":
        model = load_coverhunter_model()
        def similarity_function(a, b):
            return compute_similarity_coverhunter(a, b, model)
    elif model_name in ["MFCC", "Spectral Centroid"]:
        def similarity_function(a, b):
            return compute_similarity(a, b, model_name)
        model = None
    else:
        raise ValueError("Unsupported model. Choose ByteCover, CoverHunter, MFCC, or Spectral Centroid.")

    # Gather and compute
    files_with_ground_truth = gather_injected_abracadabra_files(dataset_path, ground_truth_file)
    results = compute_injected_abracadabra_results(
        files_with_ground_truth, reference_song, similarity_function, threshold, progress
    )

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
        "Total Files (Ground Truth Covers)": total_relevant,
        "Threshold Used": threshold
    }

    return summary_metrics

