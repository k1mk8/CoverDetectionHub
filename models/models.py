import os
import sys
import torch


# Add CoverHunter directory to sys.path
coverhunter_path = os.path.abspath("./models/CoverHunter")
if coverhunter_path not in sys.path:
    sys.path.insert(0, coverhunter_path)


import torchaudio
from models.bytecover.bytecover.models.modules import Bottleneck, Resnet50
from models.CoverHunter.src.model import Model
from models.CoverHunter.src.utils import load_hparams
from scipy.spatial.distance import cosine
from models.CoverHunter.src.cqt import PyCqt  # Assuming PyCqt is available in the CoverHunter repo
import numpy as np
import torchaudio

# Configuration
BYTECOVER_CHECKPOINT_PATH = "models/bytecover/models/orfium-bytecover.pt"
TARGET_SR = 22050
MAX_LEN = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ByteCover model
def load_bytecover_model(checkpoint_path=BYTECOVER_CHECKPOINT_PATH):
    model = Resnet50(
        Bottleneck, num_channels=1, num_classes=10000, compress_ratio=20, tempo_factors=[0.7, 1.3]
    )
    model.to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model

# Preprocess audio
def preprocess_audio(file_path, target_sr=TARGET_SR, max_len=MAX_LEN):
    waveform, sr = torchaudio.load(file_path)
    resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
    waveform = resample(waveform)

    if waveform.size(0) > 1:  # Convert to mono
        waveform = waveform.mean(dim=0, keepdim=True)

    max_samples = target_sr * max_len
    if waveform.size(1) > max_samples:
        waveform = waveform[:, :max_samples]
    else:
        pad = max_samples - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, pad))

    return waveform.squeeze(0)  # Return 1D tensor

# Compute similarity with ByteCover
def compute_similarity_bytecover(song1_path, song2_path, model):
    song1 = preprocess_audio(song1_path).unsqueeze(0).to(DEVICE)  # Add batch dimension
    song2 = preprocess_audio(song2_path).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features1 = model(song1)
        features2 = model(song2)

    embedding1 = features1["f_t"]
    embedding2 = features2["f_t"]

    similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
    return similarity.item()


# Configuration
COVERHUNTER_CONFIG_PATH = "./models/CoverHunter/pretrain_model/config/hparams.yaml"
COVERHUNTER_CHECKPOINT_DIR = "./models/CoverHunter/pretrain_model/pt_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CoverHunter Model
def load_coverhunter_model(config_path=COVERHUNTER_CONFIG_PATH, checkpoint_dir=COVERHUNTER_CHECKPOINT_DIR):
    """
    Load the CoverHunter model using its custom load_model_parameters method.
    Dynamically handle device (CPU or CUDA).
    """
    # Load hyperparameters
    hp = load_hparams(config_path)

    # Initialize the model
    model = Model(hp)

    # Ensure checkpoint directory exists
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Load model parameters
    epoch = model.load_model_parameters(checkpoint_dir)

    # Move the model to the appropriate device
    model = model.to(DEVICE)
    print(f"CoverHunter model loaded from epoch {epoch} on {DEVICE}")

    return model


# Preprocess Audio
def preprocess_audio_coverhunter(file_path, target_sr=16000, max_len=100):
    """
    Preprocess an audio file for CoverHunter by resampling, padding, and extracting CSI features.

    Args:
        file_path (str): Path to the audio file.
        target_sr (int): Target sample rate for the audio file.
        max_len (int): Maximum length of the audio in seconds.

    Returns:
        torch.Tensor: CSI features with shape [1, frame_size, feat_size].
    """
    # Step 1: Load and Resample Audio
    waveform, sr = torchaudio.load(file_path)
    resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
    waveform = resample(waveform)

    # Step 2: Convert to Mono
    if waveform.size(0) > 1:  # If stereo, average channels to create mono
        waveform = waveform.mean(dim=0, keepdim=True)

    # Step 3: Trim or Pad Audio to max_len
    max_samples = target_sr * max_len
    if waveform.size(1) > max_samples:
        waveform = waveform[:, :max_samples]
    else:
        pad = max_samples - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, pad))

    # Step 4: Normalize Audio
    waveform_np = waveform.squeeze(0).numpy()  # Convert to NumPy for compatibility with PyCqt
    waveform_np = waveform_np / max(0.001, np.max(np.abs(waveform_np))) * 0.999

    # Step 5: Extract CSI Features using PyCqt
    py_cqt = PyCqt(
        sample_rate=target_sr,
        hop_size=0.04,  # Match hop_size in hparams.yaml
        octave_resolution=12,  # Adjust bins per octave if needed
        min_freq=32,  # Adjust based on the dataset configuration
        max_freq=target_sr // 2  # Nyquist frequency
    )
    csi_features = py_cqt.compute_cqt(signal_float=waveform_np, feat_dim_first=False)
    print(f"CQT spectrogram shape: {csi_features.shape}")  # Expect [frame_size, 101]

    # Step 6: Add Batch Dimension
    csi_tensor = torch.tensor(csi_features, dtype=torch.float32).unsqueeze(0)  # Shape: [1, frame_size, feat_size]

    return csi_tensor.to(DEVICE)

# Compute Similarity with CoverHunter
def compute_similarity_coverhunter(audio1_path, audio2_path, model):
    """
    Compute similarity between two audio files using the CoverHunter model.

    Args:
        audio1_path (str): Path to the first audio file.
        audio2_path (str): Path to the second audio file.
        model: The loaded CoverHunter model.

    Returns:
        float: Similarity score between the two audio files.
    """
    # Preprocess audio files to CSI features
    features1 = preprocess_audio_coverhunter(audio1_path).to(DEVICE)
    features2 = preprocess_audio_coverhunter(audio2_path).to(DEVICE)

    # Pass features through the model to obtain embeddings
    with torch.no_grad():
        embedding1 = model(features1)
        embedding2 = model(features2)

    # Compute cosine similarity
    similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
    return similarity.item()

# Configuration
REMOVE_CONFIG_PATH = "models/re-move/data/baseline_defaults.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")