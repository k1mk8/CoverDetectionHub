from abc import ABC, abstractmethod
import torch


class ModelBase(ABC):
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    @abstractmethod
    def _load_model(self):
        """Load the model with its specific parameters."""
        pass

    @abstractmethod
    def compute_similarity_between_files(self, audio1_path, audio2_path):
        """Compute similarity between two audio files."""
        pass

    @abstractmethod
    def compute_embedding(self, audio_path):
        """Compute embeddings for a given audio file."""
        pass

    @abstractmethod
    def compute_similarity(self, embedding1, embedding2):
        """Compute similarity between two embeddings."""
        pass
