
import os
import sys
import torch
from feature_extraction.audio_preprocessing import preprocess_audio_coverhunter

# Add CoverHunter directory to sys.path
coverhunter_path = os.path.abspath("./csi_models/CoverHunter")
if coverhunter_path not in sys.path:
    sys.path.insert(0, coverhunter_path)


from csi_models.bytecover.bytecover.models.modules import Bottleneck, Resnet50
from csi_models.CoverHunter.src.model import Model
from csi_models.CoverHunter.src.utils import load_hparams
from scipy.spatial.distance import cosine
from csi_models.CoverHunter.src.cqt import PyCqt  # Assuming PyCqt is available in the CoverHunter repo
import numpy as np
import torchaudio


# Configuration
# COVERHUNTER_CONFIG_PATH = "./models/checkpoints/CoverHunter/pretrain_model/config/hparams.yaml"
# COVERHUNTER_CHECKPOINT_DIR = "./models/checkpoints/CoverHunter/pretrain_model/pt_model"
COVERHUNTER_CONFIG_PATH = "./csi_models/checkpoints/CoverHunter/our_model/config/hparams.yaml"
COVERHUNTER_CHECKPOINT_DIR = "./csi_models/checkpoints/CoverHunter/our_model/pt_model"
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