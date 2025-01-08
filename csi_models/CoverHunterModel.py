import os
import sys
import torch
import yaml

from feature_extraction.audio_preprocessing import preprocess_audio_coverhunter
from csi_models.ModelBase import ModelBase

coverhunter_path = os.path.abspath("./csi_models/CoverHunter")
if coverhunter_path not in sys.path:
    sys.path.insert(0, coverhunter_path)


from csi_models.CoverHunter.src.model import Model
from csi_models.CoverHunter.src.utils import load_hparams


class CoverHunterModel(ModelBase):
    def __init__(self, config_path: str = "configs/paths.yaml", device: torch.device = None):
        # Load configuration
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.config_path = config["coverhunter_config_path"]
        self.checkpoint_dir = config["coverhunter_checkpoint_dir"]
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Add CoverHunter directory to sys.path
        coverhunter_path = os.path.abspath("./csi_models/CoverHunter")
        if coverhunter_path not in sys.path:
            sys.path.insert(0, coverhunter_path)

        # Load the model
        self.model = self._load_model()

    def _load_model(self) -> Model:
        # Load hyperparameters
        hp = load_hparams(self.config_path)

        # Initialize the model
        model = Model(hp)

        # Ensure checkpoint directory exists
        if not os.path.exists(self.checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {self.checkpoint_dir}")

        # Load model parameters
        epoch = model.load_model_parameters(self.checkpoint_dir)

        # Move the model to the appropriate device
        model = model.to(self.device)
        print(f"CoverHunter model loaded from epoch {epoch} on {self.device}")

        return model

    def compute_similarity_between_files(self, audio1_path: str, audio2_path: str) -> float:
        # Preprocess audio files to CSI features
        features1 = preprocess_audio_coverhunter(audio1_path).to(self.device)
        features2 = preprocess_audio_coverhunter(audio2_path).to(self.device)

        # Pass features through the model to obtain embeddings
        with torch.no_grad():
            embedding1 = self.model(features1)
            embedding2 = self.model(features2)

        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
        return similarity.item()

    def compute_embedding(self, audio_path: str) -> torch.Tensor:
        # Preprocess audio file to CSI features
        features = preprocess_audio_coverhunter(audio_path).to(self.device)

        # Pass features through the model to obtain embeddings
        with torch.no_grad():
            embeddings = self.model(features)

        return embeddings

    def compute_similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
        return similarity.item()
