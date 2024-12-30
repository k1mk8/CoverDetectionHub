import os
import sys
import torch


# Add CoverHunter directory to sys.path
coverhunter_path = os.path.abspath("./models/CoverHunter")
if coverhunter_path not in sys.path:
    sys.path.insert(0, coverhunter_path)


from models.bytecover.bytecover.models.modules import Bottleneck, Resnet50
from models.CoverHunter.src.model import Model
from models.CoverHunter.src.utils import load_hparams
from scipy.spatial.distance import cosine
from models.CoverHunter.src.cqt import PyCqt  # Assuming PyCqt is available in the CoverHunter repo
import numpy as np
import torchaudio

from preprocessing import preprocess_audio, preprocess_audio_coverhunter

# Configuration
# BYTECOVER_CHECKPOINT_PATH = "models/checkpoints/bytecover/orfium-bytecover.pt"
BYTECOVER_CHECKPOINT_PATH = "models/checkpoints/bytecover/bytecover_run4.pt"
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

# Compute similarity with ByteCover
def compute_similarity_bytecover(song1_path, song2_path, model):
    song1 = preprocess_audio(song1_path).unsqueeze(0).to(DEVICE)  # Add batch dimension
    song2 = preprocess_audio(song2_path).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features1 = model(song1)
        features2 = model(song2)

    # Extract and normalize embeddings
    embedding1 = torch.nn.functional.normalize(features1["f_t"], p=2, dim=1)
    embedding2 = torch.nn.functional.normalize(features2["f_t"], p=2, dim=1)

    # Compute cosine similarity
    similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
    return similarity.item()


# Configuration
# COVERHUNTER_CONFIG_PATH = "./models/checkpoints/CoverHunter/pretrain_model/config/hparams.yaml"
# COVERHUNTER_CHECKPOINT_DIR = "./models/checkpoints/CoverHunter/pretrain_model/pt_model"
COVERHUNTER_CONFIG_PATH = "./models/checkpoints/CoverHunter/our_model/config/hparams.yaml"
COVERHUNTER_CHECKPOINT_DIR = "./models/checkpoints/CoverHunter/our_model/pt_model"
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
REMOVE_CHECKPOINT_DIR = ""
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Compute Similarity with Re-move
def compute_similarity_remove(audio1_path, audio2_path, model):
    """
    Compute similarity between two audio files using the re-move model.

    Args:
        audio1_path (str): Path to the first audio file.
        audio2_path (str): Path to the second audio file.
        model: The loaded re-move model.

    Returns:
        float: Similarity score between the two audio files.
    """
    # Preprocess audio files to CSI features
    features1 = preprocess_audio(audio1_path).to(DEVICE)
    features2 = preprocess_audio(audio2_path).to(DEVICE)

    # Pass features through the model to obtain embeddings
    with torch.no_grad():
        embedding1 = model(features1)
        embedding2 = model(features2)

    # Compute cosine similarity
    similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
    return similarity.item()


def compute_batch_similarity_bytecover(batch_a_paths, batch_b_paths, model):
    batch_a = torch.stack([preprocess_audio(path).unsqueeze(0) for path in batch_a_paths]).to(DEVICE)
    batch_b = torch.stack([preprocess_audio(path).unsqueeze(0) for path in batch_b_paths]).to(DEVICE)

    with torch.no_grad():
        features_a = model(batch_a)
        features_b = model(batch_b)

    embeddings_a = features_a["f_t"]
    embeddings_b = features_b["f_t"]

    # Compute cosine similarity for each pair in the batch
    similarities = torch.nn.functional.cosine_similarity(embeddings_a, embeddings_b)
    return similarities.cpu().tolist()

# Batch processing for CoverHunter

def compute_batch_similarity_coverhunter(batch_a_paths, batch_b_paths, model):
    batch_a = torch.stack([preprocess_audio_coverhunter(path) for path in batch_a_paths]).to(DEVICE)
    batch_b = torch.stack([preprocess_audio_coverhunter(path) for path in batch_b_paths]).to(DEVICE)

    with torch.no_grad():
        embeddings_a = model(batch_a)
        embeddings_b = model(batch_b)

    # Compute cosine similarity for each pair in the batch
    similarities = torch.nn.functional.cosine_similarity(embeddings_a, embeddings_b)
    return similarities.cpu().tolist()
