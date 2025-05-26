import os
import torch
import torchaudio
import whisper
from audiocraft.models import MusicGen
from bark import generate_audio, preload_models
from TTS.api import TTS

if not hasattr(torch, "get_default_device"):
    torch.get_default_device = lambda: "cuda" if torch.cuda.is_available() else "cpu"

_musicgen_model = None
_tts_model = None
_whisper_model_cache = {}

def _load_audio(path: str, target_sr: int) -> torch.Tensor:
    """
    Loads an audio file and resamples it to the target sample rate.
    Converts stereo to mono by averaging channels if necessary.
    """
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav

def _trim_or_pad(wav: torch.Tensor, duration_s: int, sr: int) -> torch.Tensor:
    """
    Trims or pads the audio tensor to a specific duration in seconds.
    """
    max_len = duration_s * sr
    length = wav.shape[-1]
    if length > max_len:
        return wav[..., :max_len]
    elif length < max_len:
        return torch.nn.functional.pad(wav, (0, max_len - length))
    return wav

def _get_musicgen(device: str = None) -> MusicGen:
    """
    Loads and returns the MusicGen model instance.
    Caches the model to avoid repeated loading.
    """
    global _musicgen_model
    if _musicgen_model is None:
        dev = device or torch.get_default_device()
        _musicgen_model = MusicGen.get_pretrained("facebook/musicgen-melody", device=dev)
    return _musicgen_model

def _get_whisper(name: str = "small") -> whisper.Whisper:
    """
    Loads and returns the Whisper model instance with a given size.
    Uses caching to prevent re-loading.
    """
    if name not in _whisper_model_cache:
        _whisper_model_cache[name] = whisper.load_model(name)
    return _whisper_model_cache[name]

def _get_tts() -> TTS:
    """
    Loads and returns the TTS singing model.
    Caches the model for reuse.
    """
    global _tts_model
    if _tts_model is None:
        _tts_model = TTS(
            model_name="singing/vits/vctk/vits_singing",
            progress_bar=False,
            gpu=torch.cuda.is_available(),
        )
    return _tts_model

def generate_cover(audio_path: str, duration: int) -> str:
    """
    Generates an instrumental cover of a given audio file using MusicGen.
    Returns the path to the generated audio file or an error message.
    """
    device = torch.get_default_device()
    try:
        music_model = _get_musicgen(device)
        wav_mel = _load_audio(audio_path, music_model.sample_rate)
        wav_mel = _trim_or_pad(wav_mel, duration, music_model.sample_rate)
        music_model.set_generation_params(duration=duration)
        inst = music_model.generate_with_chroma([""], wav_mel, music_model.sample_rate)[0]
        if inst.dim() == 3:
            inst = inst.squeeze(0)
        out_path = "generated_cover.wav"
        torchaudio.save(out_path, inst.cpu(), sample_rate=music_model.sample_rate)
        return out_path
    except Exception as e:
        return f"Error generating instrumental cover: {e}"

def generate_cover_with_lyrics(
    audio_path: str,
    duration: int,
    cover_type: str,
    whisper_model: str = "small",
    vocal_gain: float = 2,
    singer_gender: str = "Female",
    tmp_dir: str = "/tmp",
) -> str:
    """
    Generates a vocal cover of the input audio with lyrics.
    
    Steps:
    - Extracts instrumental using MusicGen.
    - Transcribes lyrics using Whisper.
    - Synthesizes vocal with Bark using transcribed lyrics.
    - Mixes vocal with instrumental and normalizes volume.
    
    Returns the path to the final cover audio file.
    
    Parameters:
    - audio_path: Path to input audio file.
    - duration: Target duration in seconds.
    - cover_type: Unused (placeholder for future logic).
    - whisper_model: Whisper model variant to use for transcription.
    - vocal_gain: Gain multiplier applied to synthesized vocals.
    - singer_gender: Unused (placeholder).
    - tmp_dir: Temporary directory to save intermediate files.
    """
    device = torch.get_default_device()

    music_model = _get_musicgen(device)
    sr = music_model.sample_rate
    wav = _load_audio(audio_path, sr)
    wav = _trim_or_pad(wav, duration, sr)

    music_model.set_generation_params(duration=duration)
    inst = music_model.generate_with_chroma([""], wav.unsqueeze(0), sr)[0]
    if inst.dim() == 3:
        inst = inst.squeeze(0)

    wm = _get_whisper(whisper_model)
    wav16 = torchaudio.transforms.Resample(sr, whisper.audio.SAMPLE_RATE)(wav)
    tmp_wav = os.path.join(tmp_dir, "whisp_in.wav")
    torchaudio.save(tmp_wav, wav16, whisper.audio.SAMPLE_RATE)
    lyrics = wm.transcribe(tmp_wav).get("text", "").strip()

    bark_wav = generate_audio(text=lyrics if lyrics else "♪ vocal melody ♪")
    voc = torch.from_numpy(bark_wav).unsqueeze(0)
    voc_sr = 24000
    voc = _trim_or_pad(voc, duration, voc_sr)
    if voc_sr != sr:
        voc = torchaudio.transforms.Resample(voc_sr, sr)(voc)
    voc = voc * vocal_gain
    voc = voc.to(device)

    length = max(inst.shape[-1], voc.shape[-1])
    inst = torch.nn.functional.pad(inst, (0, length - inst.shape[-1]))
    voc = torch.nn.functional.pad(voc, (0, length - voc.shape[-1]))
    mix = inst + voc
    mix = mix / mix.abs().max()

    mix = mix.cpu()
    if mix.dim() == 1:
        mix = mix.unsqueeze(0)

    out_path = "cover_with_lyrics_bark_gpu.wav"
    torchaudio.save(out_path, mix, sample_rate=sr)
    return out_path