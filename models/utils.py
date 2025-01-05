import numpy as np
from scipy.spatial.distance import cosine
import librosa
import random
import gradio as gr

import os

from models.models import compute_batch_similarity_bytecover, compute_batch_similarity_coverhunter, compute_similarity_bytecover, compute_similarity_coverhunter, load_bytecover_model, load_coverhunter_model

# Load audio for other models
def load_audio(filepath, sr=22050):
    y, sr = librosa.load(filepath, sr=sr)
    return y

# Feature extraction for Model 1
def extract_features_model_1(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=13)
    return np.mean(mfcc, axis=1)

# Feature extraction for Model 2
def extract_features_model_2(audio):
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=22050)
    return np.mean(spectral_centroid, axis=1)

# Determine if two songs are covers using the chosen model
def is_cover_song(audio1_path, audio2_path, model_name, bytecover_model=None, threshold=0.99):
    if model_name == "ByteCover":
        similarity = compute_similarity_bytecover(audio1_path, audio2_path, bytecover_model)
        return ("Cover", similarity) if similarity >= threshold else ("Not a Cover", similarity)

    audio1 = load_audio(audio1_path)
    audio2 = load_audio(audio2_path)
    features1 = extract_features_model_1(audio1) if model_name == "MFCC" else extract_features_model_2(audio1)
    features2 = extract_features_model_1(audio2) if model_name == "MFCC" else extract_features_model_2(audio2)
    similarity_score = 1 - cosine(features1, features2)
    return ("Cover", similarity_score) if similarity_score >= threshold else ("Not a Cover", similarity_score)

# Dummy dataset evaluation function
def test_model_on_dataset(model_name, dataset):
    random.seed(42)
    mAP = random.uniform(0.5, 0.9)
    p_at_10 = random.uniform(0.5, 0.9)
    mr1 = random.randint(1, 10)
    return {
        "Mean Average Precision (mAP)": mAP,
        "Precision at 10 (P@10)": p_at_10,
        "Mean Rank of First Correct Cover (MR1)": mr1,
    }

# Helper functions for MFCC and Spectral Centroid models
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

def evaluate_on_covers80(model_name, threshold=0.99, covers80but10=False, progress=gr.Progress()):
    if covers80but10:
        dataset_path = "datasets/covers80but10/coversongs/covers32k/"
    else:
        dataset_path = "datasets/covers80/coversongs/covers32k/"
    results = []

    if model_name == "ByteCover":
        model = load_bytecover_model()
        similarity_function = compute_similarity_bytecover  # No batching
    elif model_name == "CoverHunter":
        model = load_coverhunter_model()
        similarity_function = compute_similarity_coverhunter  # No batching
    elif model_name in ["MFCC", "Spectral Centroid"]:
        similarity_function = lambda a, b, _: compute_similarity(a, b, model_name)
        model = None  # Not used for MFCC or Spectral Centroid
    else:
        raise ValueError("Unsupported model. Choose ByteCover, CoverHunter, MFCC, or Spectral Centroid.")

    # Collect all audio files with their respective song labels
    all_audio_files = []
    song_labels = []
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue

        audio_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".mp3")]
        if len(audio_files) < 2:
            print(f"Skipping folder {folder} with less than 2 files.")
            continue

        all_audio_files.extend(audio_files)
        song_labels.extend([folder] * len(audio_files))

    # Prepare all pairs for non-batched processing
    num_files = len(all_audio_files)
    file_pairs = [
        (all_audio_files[i], all_audio_files[j], song_labels[i], song_labels[j])
        for i in range(num_files) for j in range(i + 1, num_files)
    ]

    total_comparisons = len(file_pairs)
    progress(0, desc="Initializing pairwise comparisons")

    for idx, (song_a, song_b, label_a, label_b) in enumerate(file_pairs):
        # Compute similarity for each pair
        similarity = similarity_function(song_a, song_b, model)

        is_cover = similarity >= threshold
        ground_truth = (label_a == label_b)  # Same song => true cover

        # Print comparison details
        print(f"Comparison {idx + 1}/{total_comparisons}")
        print(f"  Song A: {song_a}")
        print(f"  Song B: {song_b}")
        print(f"  Similarity: {similarity:.4f}")
        print(f"  Predicted Cover: {'Yes' if is_cover else 'No'}")
        print(f"  Ground Truth Cover: {'Yes' if ground_truth else 'No'}\n")

        results.append({
            "song_pair": (song_a, song_b),
            "similarity": similarity,
            "is_cover": is_cover,       # Predicted label
            "ground_truth": ground_truth
        })

        progress((idx + 1) / total_comparisons, desc="Evaluating pairs")

    # Sort by similarity (descending)
    results.sort(key=lambda x: x["similarity"], reverse=True)

    # Precision@10
    top_10 = results[:10]
    p_at_10 = sum(1 for r in top_10 if r["is_cover"] == r["ground_truth"]) / len(top_10) if len(top_10) > 0 else 0

    # Mean Rank of First Correct Cover (MR1)
    first_correct_ranks = [i + 1 for i, r in enumerate(results) if r["is_cover"] and r["ground_truth"]]
    mr1 = sum(first_correct_ranks) / len(first_correct_ranks) if first_correct_ranks else float('inf')

    # mAP Calculation
    total_relevant = sum(1 for r in results if r["ground_truth"])
    predicted_correct_count = 0
    precision_sum = 0

    for i, r in enumerate(results):
        if r["is_cover"] and r["ground_truth"]:  # Correctly predicted as cover
            predicted_correct_count += 1
            precision_at_i = predicted_correct_count / (i + 1)
            precision_sum += precision_at_i

    if total_relevant > 0:
        mAP = precision_sum / total_relevant
    else:
        mAP = 0

    summary_metrics = {
        "Mean Average Precision (mAP)": mAP,
        "Precision at 10 (P@10)": p_at_10,
        "Mean Rank of First Correct Cover (MR1)": mr1,
        "Total Covers Predicted Correctly": predicted_correct_count,
        "Total Pairs (Ground Truth Covers)": total_relevant,
        "Threshold Used": threshold
    }

    return summary_metrics
