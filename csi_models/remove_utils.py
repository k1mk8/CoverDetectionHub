import torch

from feature_extraction.audio_preprocessing import preprocess_audio
# Configuration
REMOVE_CONFIG_PATH = "csi_models/re-move/data/baseline_defaults.json"
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
