import os
import sys
import logging
import torch
import yaml

lyricover_path = os.path.abspath("./csi_models/lyricover")
if lyricover_path not in sys.path:
    sys.path.insert(0, lyricover_path)

from csi_models.lyricover.utils import load_whisper_model
from csi_models.lyricover.model import CoverClassifier

with open("configs/paths.yaml", "r") as f:
    config = yaml.safe_load(f)

LYRICOVER_CHECKPOINT_PATH = config["lyricover_checkpoint_path"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_lyricover_model(
    instrumental_threshold: int = 8,
    lyrics_model=None,
    checkpoint_path: str = LYRICOVER_CHECKPOINT_PATH
) -> CoverClassifier:
    """
    Load and initialize the Lyricover model (CoverClassifier).
    """
    logging.info("Loading Lyricover model...")

    if lyrics_model is None:
        logging.info("Loading default Whisper model...")
        lyrics_model = load_whisper_model()

    classifier = CoverClassifier(
        instrumental_threshold=instrumental_threshold,
        lyrics_model=lyrics_model
    )


    classifier.load_model(checkpoint_path)
    logging.info("Lyricover model loaded and ready for inference.")

    return classifier


def compute_similarity_lyricover(
    audio1_path: str,
    audio2_path: str,
    classifier: CoverClassifier
) -> float:
    """
    Compute a 'cover likelihood' score for two audio files using the Lyricover model.
    """
    logging.info(f"Computing Lyricover similarity for:\n  {audio1_path}\n  {audio2_path}")

    score = classifier.predict(audio1_path, audio2_path)

    logging.info(f"Lyricover cover score: {score:.4f}")
    return score
