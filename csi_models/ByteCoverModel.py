import torch
from csi_models.bytecover.bytecover.models.modules import Bottleneck, Resnet50
from csi_models.ModelBase import ModelBase
from feature_extraction.audio_preprocessing import preprocess_audio
import yaml

class ByteCoverModel(ModelBase):
    def __init__(self, config_path="configs/paths.yaml", device=None):
        # Load configuration
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        self.checkpoint_path = config["bytecover_checkpoint_path"]
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model
        self.model = self._load_model()

    def _load_model(self):
        loaded_weights = torch.load(self.checkpoint_path, map_location=self.device)
        model = Resnet50(
            Bottleneck, num_channels=1, num_classes=10000, compress_ratio=20, tempo_factors=[0.7, 1.3]
        )
        model.to(self.device)
        model.load_state_dict(loaded_weights)
        model.eval()
        return model

    def compute_similarity_between_files(self, song1_path, song2_path):
        song1 = preprocess_audio(song1_path).unsqueeze(0).to(self.device)  # Add batch dimension
        song2 = preprocess_audio(song2_path).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features1 = self.model(song1)
            features2 = self.model(song2)

        # Extract and normalize embeddings
        embedding1 = torch.nn.functional.normalize(features1["f_c"], p=2, dim=1)
        embedding2 = torch.nn.functional.normalize(features2["f_c"], p=2, dim=1)

        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
        return similarity.item()

    def compute_embedding(self, song_path):
        song = preprocess_audio(song_path).unsqueeze(0).to(self.device)  # Add batch dimension

        with torch.no_grad():
            features = self.model(song)

        # Extract and normalize embeddings
        embedding = torch.nn.functional.normalize(features["f_c"], p=2, dim=1)
        return embedding

    def compute_similarity(self, embedding1, embedding2):
        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
        return similarity.item()

# Example usage:
# bytecover = ByteCoverModel()
# similarity = bytecover.compute_similarity_between_files("song1.wav", "song2.wav")
# embedding = bytecover.compute_embedding("song1.wav")
