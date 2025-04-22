import torchaudio
import torch
from audiocraft.models import MusicGen

def generate_cover(audio_path, duration):
    """
    Generate a musical cover using MusicGen based on a melody from an input audio file.

    Args:
        audio_path (str): Path to the uploaded input melody audio file (e.g., a short music clip).
        duration (int): Desired duration (in seconds) of the generated cover song.

    Returns:
        str: Path to the saved generated audio cover (.wav), or an error message if generation fails.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        musicgen_model = MusicGen.get_pretrained('facebook/musicgen-melody', device=device)
        wav, sr = torchaudio.load(audio_path)
        if sr != musicgen_model.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=musicgen_model.sample_rate)
            wav = resampler(wav)
        if wav.dim() == 2:
            wav = wav.unsqueeze(0)
        musicgen_model.set_generation_params(duration=duration)
        out_wav = musicgen_model.generate_with_chroma(
            descriptions=[""], melody_wavs=wav, melody_sample_rate=musicgen_model.sample_rate
        )[0]

        output_path = "generated_cover.wav"
        torchaudio.save(output_path, out_wav.cpu(), sample_rate=musicgen_model.sample_rate)

        return output_path

    except Exception as e:
        return f"Błąd: {e}"