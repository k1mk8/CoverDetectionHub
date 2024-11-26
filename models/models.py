import torch
import torchaudio
from models.bytecover.bytecover.models.modules import Bottleneck, Resnet50

# Configuration
CHECKPOINT_PATH = "models/bytecover/models/orfium-bytecover.pt"
TARGET_SR = 22050
MAX_LEN = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ByteCover model
def load_bytecover_model(checkpoint_path=CHECKPOINT_PATH):
    model = Resnet50(
        Bottleneck, num_channels=1, num_classes=10000, compress_ratio=20, tempo_factors=[0.7, 1.3]
    )
    model.to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model

# Preprocess audio
def preprocess_audio(file_path, target_sr=TARGET_SR, max_len=MAX_LEN):
    waveform, sr = torchaudio.load(file_path)
    resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
    waveform = resample(waveform)

    if waveform.size(0) > 1:  # Convert to mono
        waveform = waveform.mean(dim=0, keepdim=True)

    max_samples = target_sr * max_len
    if waveform.size(1) > max_samples:
        waveform = waveform[:, :max_samples]
    else:
        pad = max_samples - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, pad))

    return waveform.squeeze(0)  # Return 1D tensor

# Compute similarity with ByteCover
def compute_similarity_bytecover(song1_path, song2_path, model):
    song1 = preprocess_audio(song1_path).unsqueeze(0).to(DEVICE)  # Add batch dimension
    song2 = preprocess_audio(song2_path).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features1 = model(song1)
        features2 = model(song2)

    embedding1 = features1["f_t"]
    embedding2 = features2["f_t"]

    similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
    return similarity.item()
