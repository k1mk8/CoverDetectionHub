# feature_extraction/feature_extraction.py

import numpy as np
import librosa
from csi_models.CoverHunter.src.cqt import PyCqt
from scipy.spatial.distance import cosine

def load_audio(filepath, sr=22050):
    y, sr = librosa.load(filepath, sr=sr)
    return y

def extract_features_mfcc(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=13)
    return np.mean(mfcc, axis=1)

def extract_features_spectral_centroid(audio):
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=22050)
    return np.mean(spectral_centroid, axis=1)

def extract_features_cqt(
    audio_np: np.ndarray,
    sample_rate: int,
    hop_size: float = 0.04,
    octave_resolution: int = 12,
    min_freq: float = 32.0,
    max_freq: float = None
):
    """
    Extract CQT-based features from a numpy audio signal using PyCqt.
    """
    if max_freq is None:
        max_freq = sample_rate // 2  # Nyquist frequency by default
    
    # Instantiate PyCqt with your desired parameters
    py_cqt = PyCqt(
        sample_rate=sample_rate,
        hop_size=hop_size,
        octave_resolution=octave_resolution,
        min_freq=min_freq,
        max_freq=max_freq
    )

    # Compute the CQT features
    cqt_features = py_cqt.compute_cqt(signal_float=audio_np, feat_dim_first=False)
    return cqt_features


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

