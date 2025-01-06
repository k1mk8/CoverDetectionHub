import os
import random
import logging
import numpy as np
import pandas as pd
import librosa
import gradio as gr
from scipy.spatial.distance import cosine

from models.models import (
    compute_batch_similarity_bytecover,
    compute_batch_similarity_coverhunter,
    compute_similarity_bytecover,
    compute_similarity_coverhunter,
    load_bytecover_model,
    load_coverhunter_model
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("cover_song_evaluation.log")
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

def load_audio(filepath, sr=22050):
    y, sr = librosa.load(filepath, sr=sr)
    return y

def extract_features_mfcc(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=13)
    return np.mean(mfcc, axis=1)

def extract_features_spectral_centroid(audio):
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=22050)
    return np.mean(spectral_centroid, axis=1)

def compute_similarity(audio1_path, audio2_path, model_name):
    """
    Computes similarity for MFCC or Spectral Centroid only.
    """
    audio1 = load_audio(audio1_path)
    audio2 = load_audio(audio2_path)

    if model_name == "MFCC":
        features1 = extract_features_mfcc(audio1)
        features2 = extract_features_mfcc(audio2)
    elif model_name == "Spectral Centroid":
        features1 = extract_features_spectral_centroid(audio1)
        features2 = extract_features_spectral_centroid(audio2)
    else:
        raise ValueError("Unsupported model for similarity computation")

    return 1 - cosine(features1, features2)


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
    Evaluate any of the four models (ByteCover, CoverHunter, MFCC, Spectral Centroid)
    on the covers80 or covers80but10 dataset.
    """
    if covers80but10:
        dataset_path = "datasets/covers80but10/coversongs/covers32k/"
    else:
        dataset_path = "datasets/covers80/coversongs/covers32k/"

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
        # We pass model_name in closure
        def similarity_function(a, b):
            return compute_similarity(a, b, model_name)
        model = None
    else:
        raise ValueError("Unsupported model. Choose ByteCover, CoverHunter, MFCC, or Spectral Centroid.")

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
        # relative path for dictionary lookup
        rel_path = os.path.relpath(path, start="datasets/injected_abracadabra").replace("\\", "/")
        files_with_ground_truth.append((path, injection_dict.get(rel_path, False)))

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
    dataset_path = "datasets/injected_abracadabra/mix/"
    ground_truth_file = "datasets/injected_abracadabra/injection_list.csv"
    reference_song = "datasets/injected_abracadabra/steve_miller_band+Steve_Miller_Band_Live_+09-Abracadabra.mp3.wav"

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

