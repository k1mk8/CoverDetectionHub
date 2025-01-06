import librosa
import torchaudio
import torch
from csi_models.CoverHunter.src.cqt import PyCqt
import numpy as np

TARGET_SR = 22050
MAX_LEN = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Preprocess Audio
def preprocess_audio_coverhunter(file_path, target_sr=16000, max_len=100):
    """
    Preprocess an audio file for CoverHunter by resampling, padding, and extracting CSI features.

    Args:
        file_path (str): Path to the audio file.
        target_sr (int): Target sample rate for the audio file.
        max_len (int): Maximum length of the audio in seconds.

    Returns:
        torch.Tensor: CSI features with shape [1, frame_size, feat_size].
    """
    # Step 1: Load and Resample Audio
    waveform, sr = torchaudio.load(file_path)
    resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
    waveform = resample(waveform)

    # Step 2: Convert to Mono
    if waveform.size(0) > 1:  # If stereo, average channels to create mono
        waveform = waveform.mean(dim=0, keepdim=True)

    # Step 3: Trim or Pad Audio to max_len
    max_samples = target_sr * max_len
    if waveform.size(1) > max_samples:
        waveform = waveform[:, :max_samples]
    else:
        pad = max_samples - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, pad))

    # Step 4: Normalize Audio
    waveform_np = waveform.squeeze(0).numpy()  # Convert to NumPy for compatibility with PyCqt
    waveform_np = waveform_np / max(0.001, np.max(np.abs(waveform_np))) * 0.999

    # Step 5: Extract CSI Features using PyCqt
    py_cqt = PyCqt(
        sample_rate=target_sr,
        hop_size=0.04,  # Match hop_size in hparams.yaml
        octave_resolution=12,  # Adjust bins per octave if needed
        min_freq=32,  # Adjust based on the dataset configuration
        max_freq=target_sr // 2  # Nyquist frequency
    )
    csi_features = py_cqt.compute_cqt(signal_float=waveform_np, feat_dim_first=False)
    print(f"CQT spectrogram shape: {csi_features.shape}")  # Expect [frame_size, 101]

    # Step 6: Add Batch Dimension
    csi_tensor = torch.tensor(csi_features, dtype=torch.float32).unsqueeze(0)  # Shape: [1, frame_size, feat_size]

    return csi_tensor.to(DEVICE)


import ffmpeg
import os

class InvalidMediaFileError(Exception):
    """Exception raised when the input file is not a valid media file."""
    pass

def validate_audio(filepath):
    """
    Validates and preprocesses an audio file.

    Parameters:
        filepath (str): Path to the input file.

    Returns:
        tuple: (processed_filepath, error_message)
        - processed_filepath: Path to the processed audio file, or None if invalid.
        - error_message: Error message if the file is invalid, or None if valid.

    Raises:
        InvalidMediaFileError: If the file is not a valid media file.
    """
    # Probe the file using ffmpeg
    try:
        probe = ffmpeg.probe(filepath)
    except ffmpeg.Error:
        raise InvalidMediaFileError("The file is not a valid media file or cannot be processed.")

    format_info = probe.get("format", {})
    duration = float(format_info.get("duration", 0))
    size = float(format_info.get("size", 0)) / (1024 * 1024)  # Convert size to MB

    # Reject if file is too large or too long
    if size > 100:
        return None, "File is too large (over 100MB)."
    if duration > 20 * 60:  # 20 minutes
        return None, "File is too long (over 20 minutes)."

    # Check if it's already an audio file
    audio_streams = [stream for stream in probe.get("streams", []) if stream.get("codec_type") == "audio"]
    if len(audio_streams) > 0:
        return filepath, None

    # If it's a video file, extract audio and convert to WAV with sr=16k
    output_filepath = os.path.splitext(filepath)[0] + ".wav"
    ffmpeg.input(filepath).output(output_filepath, format="wav", ac=1, ar="16000").run(overwrite_output=True)
    return output_filepath, None
