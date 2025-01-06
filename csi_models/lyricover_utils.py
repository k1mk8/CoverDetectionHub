import os
import sys
import torch
import yaml
import logging
import numpy as np

# Add CoverHunter directory to sys.path
coverhunter_path = os.path.abspath("./csi_models/lyricover")
if coverhunter_path not in sys.path:
    sys.path.insert(0, coverhunter_path)

from csi_models.lyricover.model import CoverClassifier
from csi_models.lyricover.utils import (
    load_whisper_model,
    generate_lyrics,
    extract_tonal_features,
    compute_cosine_similarity
)


with open("configs/paths.yaml", "r") as f:
    config = yaml.safe_load(f)

LYRICOVER_CHECKPOINT_PATH = config["lyricover_checkpoint_path"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_lyricover_model(
    instrumental_threshold: int = 8,
    whisper_model=None,
    checkpoint_path: str = LYRICOVER_CHECKPOINT_PATH
) -> CoverClassifier:
    """
    Load and initialize a CoverClassifier.
    """
    logging.info("Loading Lyricover model...")

    # Load Whisper or other lyrics model if not supplied
    if whisper_model is None:
        logging.info("No Whisper model supplied; loading default Whisper.")
        whisper_model = load_whisper_model()

    # Create classifier
    classifier = CoverClassifier(
        instrumental_threshold=instrumental_threshold,
        lyrics_model=whisper_model
    )

    # Load the PyTorch checkpoint (if you want the NN weights)
    logging.info(f"Loading Lyricover checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    classifier.nn_model.load_state_dict(checkpoint)
    classifier.nn_model.to(DEVICE)
    classifier.nn_model.eval()

    logging.info("Lyricover model ready for inference.")
    return classifier


def compute_similarity_lyricover(
    audio_a_path: str,
    audio_b_path: str,
    classifier: CoverClassifier,
    model_path: str = LYRICOVER_CHECKPOINT_PATH
) -> float:
    """
    Compute a cover-likelihood score for two audio files using the Lyricover approach.
    """
    logging.info(f"Computing Lyricover similarity for:\n  {audio_a_path}\n  {audio_b_path}")

    cover_score = classifier.predict(
        audio_a=audio_a_path,
        audio_b=audio_b_path,
        model_path=model_path if model_path else LYRICOVER_CHECKPOINT_PATH
    )

    logging.info(f"Lyricover cover score: {cover_score:.4f}")
    return cover_score
