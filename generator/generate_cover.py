import os
import math
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
    Converts stereo to mono by averaging channels.
    """
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav


def _trim_or_pad(wav: torch.Tensor, duration_s: float, sr: int) -> torch.Tensor:
    """
    Trims or pads the audio tensor to a specific duration in seconds.
    """
    max_len = int(duration_s * sr)
    length = wav.shape[-1]
    if length > max_len:
        return wav[..., :max_len]
    if length < max_len:
        pad_amount = max_len - length
        return torch.nn.functional.pad(wav, (0, pad_amount))
    return wav


def _get_musicgen(device: str = None) -> MusicGen:
    """
    Loads and returns the MusicGen model instance. Caches the model.
    """
    global _musicgen_model
    if _musicgen_model is None:
        dev = device or torch.get_default_device()
        _musicgen_model = MusicGen.get_pretrained("facebook/musicgen-melody", device=dev)
    return _musicgen_model


def _get_whisper(name: str = "small") -> whisper.Whisper:
    """
    Loads and returns the Whisper model instance with the given size. Caches the model.
    """
    global _whisper_model_cache
    if name not in _whisper_model_cache:
        _whisper_model_cache[name] = whisper.load_model(name)
    return _whisper_model_cache[name]


def _get_tts() -> TTS:
    """
    Loads and returns the TTS singing model. Caches the model.
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


def generate_cover_with_lyrics(
    audio_path: str,
    duration: int,
    cover_type: str,
    whisper_model: str = "small",
    vocal_gain: float = 2.0,
    singer_gender: str = "Female",
    tmp_dir: str = "/tmp",
) -> str:
    """
    Generates a vocal cover with lyrics aligned to original timing.

    Steps:
    1. Extract instrumental with MusicGen.
    2. Transcribe with Whisper (timestamps).
    3. Synthesize each segment with Bark using gender preset.
    4. Place vocal segments at original timestamps.
    5. Mix and normalize.

    Returns path to the final cover audio file.
    """
    device = torch.get_default_device()

    # presets
    voice_presets = {"female": "v2/en_speaker_9", "male": "v2/en_speaker_6"}
    preset = voice_presets.get(singer_gender.lower(), "v2/en_speaker_9")

    # Instrumental
    model = _get_musicgen(device)
    sr = model.sample_rate
    wav = _load_audio(audio_path, sr)
    wav = _trim_or_pad(wav, duration, sr)
    model.set_generation_params(duration=duration)
    inst = model.generate_with_chroma([""], wav.unsqueeze(0), sr)[0]
    if inst.dim() == 3:
        inst = inst.squeeze(0)

    # Transcribe with timestamps
    whisper_obj = _get_whisper(whisper_model)
    wav16 = torchaudio.transforms.Resample(sr, whisper.audio.SAMPLE_RATE)(wav)
    tmp = os.path.join(tmp_dir, "whisp.wav")
    torchaudio.save(tmp, wav16, whisper.audio.SAMPLE_RATE)
    result = whisper_obj.transcribe(tmp, word_timestamps=True)
    segments = result.get("segments", [])

    # Prepare empty vocal timeline
    total_len = duration * sr
    voc_timeline = torch.zeros(total_len)

    # Synthesize and place segments
    for seg in segments:
        text = seg.get("text", "").strip()
        start_s = seg.get("start", 0.0)
        end_s = seg.get("end", start_s)
        bark_wav = generate_audio(text=text or "♪ melody ♪", history_prompt=preset)
        frag = torch.from_numpy(bark_wav).unsqueeze(0)
        frag = torchaudio.transforms.Resample(24000, sr)(frag)
        seg_dur = end_s - start_s
        frag = _trim_or_pad(frag, seg_dur, sr)
        start_idx = int(math.floor(start_s * sr))
        end_idx = min(start_idx + frag.shape[-1], total_len)
        voc_timeline[start_idx:end_idx] += frag.squeeze(0)

    # apply gain and to device
    voc_timeline = voc_timeline * vocal_gain
    voc = voc_timeline.unsqueeze(0).to(device)

    # Mix and normalize
    length = max(inst.shape[-1], voc.shape[-1])
    inst = torch.nn.functional.pad(inst, (0, length - inst.shape[-1]))
    voc = torch.nn.functional.pad(voc, (0, length - voc.shape[-1]))
    mix = inst.to(device) + voc
    mix = mix / mix.abs().max()

    # to CPU and save
    mix = mix.cpu()
    if mix.dim() == 1:
        mix = mix.unsqueeze(0)
    out = "cover_with_lyrics_bark_aligned.wav"
    torchaudio.save(out, mix, sample_rate=sr)
    return out
