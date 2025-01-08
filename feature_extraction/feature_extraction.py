import logging
import librosa
import numpy as np
from csi_models.ModelBase import ModelBase
from scipy.spatial.distance import cosine
from csi_models.CoverHunter.src.cqt import PyCqt

def load_audio(filepath, sr=22050):
    y, sr = librosa.load(filepath, sr=sr)
    return y


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



class MFCCModel(ModelBase):
    def __init__(self, device=None):
        super().__init__(device)

    def _load_model(self):
        logging.info("MFCCModel does not require a model to load.")

    def compute_similarity_between_files(self, audio1_path, audio2_path):
        audio1 = self.load_audio(audio1_path)
        audio2 = self.load_audio(audio2_path)
        features1 = self.extract_features_mfcc(audio1)
        features2 = self.extract_features_mfcc(audio2)
        return 1 - cosine(features1, features2)

    def compute_embedding(self, audio_path):
        audio = self.load_audio(audio_path)
        return self.extract_features_mfcc(audio)

    def compute_similarity(self, embedding1, embedding2):
        return 1 - cosine(embedding1, embedding2)

    @staticmethod
    def load_audio(filepath, sr=22050):
        y, sr = librosa.load(filepath, sr=sr)
        return y

    @staticmethod
    def extract_features_mfcc(audio):
        mfcc = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=13)
        return np.mean(mfcc, axis=1)

class SpectralCentroidModel(ModelBase):
    def __init__(self, device=None):
        super().__init__(device)

    def _load_model(self):
        logging.info("SpectralCentroidModel does not require a model to load.")

    def compute_similarity_between_files(self, audio1_path, audio2_path):
        audio1 = self.load_audio(audio1_path)
        audio2 = self.load_audio(audio2_path)
        features1 = self.extract_features_spectral_centroid(audio1)
        features2 = self.extract_features_spectral_centroid(audio2)
        return 1 - cosine(features1, features2)

    def compute_embedding(self, audio_path):
        audio = self.load_audio(audio_path)
        return self.extract_features_spectral_centroid(audio)

    def compute_similarity(self, embedding1, embedding2):
        return 1 - cosine(embedding1, embedding2)

    @staticmethod
    def load_audio(filepath, sr=22050):
        y, sr = librosa.load(filepath, sr=sr)
        return y

    @staticmethod
    def extract_features_spectral_centroid(audio):
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=22050)
        return np.mean(spectral_centroid, axis=1)

class CQTModel(ModelBase):
    def __init__(self, device=None):
        super().__init__(device)

    def _load_model(self):
        logging.info("CQTModel does not require a model to load.")

    def compute_similarity_between_files(self, audio1_path, audio2_path):
        audio1 = self.load_audio(audio1_path)
        audio2 = self.load_audio(audio2_path)
        features1 = self.extract_features_cqt(audio1, 22050)
        features2 = self.extract_features_cqt(audio2, 22050)
        return 1 - cosine(features1.flatten(), features2.flatten())

    def compute_embedding(self, audio_path):
        audio = self.load_audio(audio_path)
        return self.extract_features_cqt(audio, 22050)

    def compute_similarity(self, embedding1, embedding2):
        return 1 - cosine(embedding1.flatten(), embedding2.flatten())

    @staticmethod
    def load_audio(filepath, sr=22050):
        y, sr = librosa.load(filepath, sr=sr)
        return y

    @staticmethod
    def extract_features_cqt(
        audio_np: np.ndarray,
        sample_rate: int,
        hop_size: float = 0.04,
        octave_resolution: int = 12,
        min_freq: float = 32.0,
        max_freq: float = None
    ):
        if max_freq is None:
            max_freq = sample_rate // 2

        py_cqt = PyCqt(
            sample_rate=sample_rate,
            hop_size=hop_size,
            octave_resolution=octave_resolution,
            min_freq=min_freq,
            max_freq=max_freq
        )

        cqt_features = py_cqt.compute_cqt(signal_float=audio_np, feat_dim_first=False)
        return cqt_features

# Example usage:
# mfcc_model = MFCCModel()
# similarity_mfcc = mfcc_model.compute_similarity_between_files("audio1.wav", "audio2.wav")
# spectral_model = SpectralCentroidModel()
# similarity_spectral = spectral_model.compute_similarity_between_files("audio1.wav", "audio2.wav")
# cqt_model = CQTModel()
# similarity_cqt = cqt_model.compute_similarity_between_files("audio1.wav", "audio2.wav")
