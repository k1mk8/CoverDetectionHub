import torch

from csi_models.bytecover.bytecover.models.modules import Bottleneck, Resnet50
from feature_extraction.audio_preprocessing import preprocess_audio

# Configuration
# BYTECOVER_CHECKPOINT_PATH = "models/checkpoints/bytecover/orfium-bytecover.pt"
BYTECOVER_CHECKPOINT_PATH = "csi_models/checkpoints/bytecover/bytecover_run4.pt"
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

