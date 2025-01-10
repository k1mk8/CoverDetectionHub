import os
import sys
import torch
import yaml

remove_path = os.path.abspath("./csi_models/remove")
if remove_path not in sys.path:
    sys.path.insert(0, remove_path)

from csi_models.ModelBase import ModelBase

from remove.models.move_model import MOVEModel
from feature_extraction.audio_preprocessing import process_crema

class RemoveModel(ModelBase):
    def __init__(self, config_path: str = "configs/paths.yaml", device: torch.device = None):
        # Load configuration
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        #self.config_path = config["remove_config_path"]
        self.checkpoint_dir = config["remove_checkpoint_dir"]
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Add Remove directory to sys.path
        remove_path = os.path.abspath("./csi_models/remove")
        if remove_path not in sys.path:
            sys.path.insert(0, remove_path)

        # Load the model
        self.model = self._load_model()
        self.emb_size = 256

    def _load_model(self) -> Model:

        model = MOVEModel(emb_size=self.emb_size)

        # Ensure checkpoint directory exists
        if not os.path.exists(self.checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {self.checkpoint_dir}")

        # Load hyperparameters
        model.load_state_dict(torch.load(self.checkpoint_dir, map_location="cpu"))
        # Load model parameters
        #epoch = model.load_model_parameters(self.checkpoint_dir)

        # Move the model to the appropriate device
        model = model.to(self.device)
        print(f"Remove model loaded on {self.device}")

        return model

    def compute_similarity_between_files(self, audio1_path: str, audio2_path: str) -> float:
        # Preprocess audio files to CSI features
        features1 = process_crema(audio1_path).to(self.device)
        features2 = process_crema(audio2_path).to(self.device)

        # Pass features through the model to obtain embeddings
        with torch.no_grad():
            embedding1 = self.model(features1)
            embedding2 = self.model(features2)

        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
        return similarity.item()

    def compute_embedding(self, audio_path: str) -> torch.Tensor:
        # Preprocess audio file to CSI features
        features = process_crema(audio_path).to(self.device)

        # Pass features through the model to obtain embeddings
        with torch.no_grad():
            embeddings = self.model(features)

        return embeddings

    def compute_similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
        return similarity.item()

remove = RemoveModel()
similarity = remove.compute_similarity_between_files("datasets/example_audio/L3xPVGosADQ.m4a", "datasets/example_audio/3ahbE6bcVf8.m4a")
print(similarity)
# embedding = bytecover.compute_embedding("song1.wav")
