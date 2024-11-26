import numpy as np
from scipy.spatial.distance import cosine
import librosa
import random

from models.models import compute_similarity_bytecover

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
def is_cover_song(audio1_path, audio2_path, model_name, bytecover_model=None):
    if model_name == "ByteCover":
        
        similarity = compute_similarity_bytecover(audio1_path, audio2_path, bytecover_model)
        threshold = 0.999
        return ("Cover", similarity) if similarity >= threshold else ("Not a Cover", similarity)

    audio1 = load_audio(audio1_path)
    audio2 = load_audio(audio2_path)
    features1 = extract_features_model_1(audio1) if model_name == "Model 1" else extract_features_model_2(audio1)
    features2 = extract_features_model_1(audio2) if model_name == "Model 1" else extract_features_model_2(audio2)
    similarity_score = 1 - cosine(features1, features2)
    threshold = 0.99
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
