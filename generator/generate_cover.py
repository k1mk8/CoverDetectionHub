import os
import torch
import torchaudio
import numpy as np
import soundfile as sf
import librosa
from audiocraft.models import MusicGen
from spleeter.separator import Separator
import pyworld as pw
import gradio as gr

# Ensure torch device helper
if not hasattr(torch, "get_default_device"):
    torch.get_default_device = lambda: "cuda" if torch.cuda.is_available() else "cpu"

# Global MusicGen model cache
_musicgen_model = None

def _load_audio(path: str, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav


def _trim_or_pad(wav: torch.Tensor, duration_s: float, sr: int) -> torch.Tensor:
    max_len = int(duration_s * sr)
    length = wav.shape[-1]
    if length > max_len:
        return wav[..., :max_len]
    if length < max_len:
        pad_amount = max_len - length
        return torch.nn.functional.pad(wav, (0, pad_amount))
    return wav


def _get_musicgen(device: str = None) -> MusicGen:
    global _musicgen_model
    if _musicgen_model is None:
        dev = device or torch.get_default_device()
        _musicgen_model = MusicGen.get_pretrained("facebook/musicgen-melody", device=dev)
    return _musicgen_model


def generate_cover(audio_path: str, duration: int) -> str:
    device = torch.get_default_device()
    try:
        model = _get_musicgen(device)
        wav = _load_audio(audio_path, model.sample_rate)
        wav = _trim_or_pad(wav, duration, model.sample_rate)
        model.set_generation_params(duration=duration)
        inst = model.generate_with_chroma([""], wav.unsqueeze(0), model.sample_rate)[0]
        if inst.dim() == 3:
            inst = inst.squeeze(0)
        out_path = "generated_cover.wav"
        torchaudio.save(out_path, inst.cpu(), sample_rate=model.sample_rate)
        return out_path
    except Exception as e:
        return f"Error generating instrumental cover: {e}"


def generate_cover_with_lyrics(audio_path: str, duration: float, alpha: float) -> str:
    # 1. Load and trim original audio to desired duration
    wav, sr = torchaudio.load(audio_path)
    wav = _trim_or_pad(wav, duration, sr)
    temp_trim = '/tmp/trimmed_input.wav'
    torchaudio.save(temp_trim, wav, sample_rate=sr)

    # 2. Separate vocals
    sep = Separator('spleeter:2stems')
    sep.separate_to_file(temp_trim, '/tmp')
    base = os.path.join('/tmp', os.path.splitext(os.path.basename(temp_trim))[0])
    vocals_path = os.path.join(base, 'vocals.wav')

    # 3. Load vocals and convert to float64
    y_voc, sr_voc = librosa.load(vocals_path, sr=None)
    y_voc = y_voc.astype(np.float64)

    # 4. WORLD analysis
    _f0, t = pw.dio(y_voc, sr_voc)
    f0 = pw.stonemask(y_voc, _f0, t, sr_voc)
    sp = pw.cheaptrick(y_voc, f0, t, sr_voc)
    ap = pw.d4c(y_voc, f0, t, sr_voc)

    # 5. Formant shifting (timbre)
    n_frames, n_bins = sp.shape
    orig_bins = np.arange(n_bins)
    src_bins = orig_bins / alpha
    sp_warped = np.zeros_like(sp)
    ap_warped = np.zeros_like(ap)
    for i in range(n_frames):
        sp_warped[i] = np.interp(orig_bins, src_bins, sp[i], left=sp[i,0], right=sp[i,-1])
        ap_warped[i] = np.interp(orig_bins, src_bins, ap[i], left=ap[i,0], right=ap[i,-1])

    # 6. Synthesize vocals
    f0_c = np.ascontiguousarray(f0)
    sp_c = np.ascontiguousarray(sp_warped)
    ap_c = np.ascontiguousarray(ap_warped)
    y_synth = pw.synthesize(f0_c, sp_c, ap_c, sr_voc)

    # 7. Write output
    out_path = '/tmp/cover_with_lyrics.wav'
    sf.write(out_path, y_synth, sr_voc)

    inst_path = generate_cover(audio_path, duration)

    inst, sr = librosa.load(inst_path, sr=None)
    voc, _ = librosa.load(out_path, sr=sr)

    min_len = min(len(inst), len(voc))
    mix = (inst[:min_len]) * 0.6 + voc[:min_len]
    result = '/tmp/result.wav'
    sf.write(result, mix, sr)
    return result