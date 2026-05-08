import os
import shutil
import subprocess
import wave
import struct
import audioop
from typing import List, Optional
from src.config.constants import SAMPLE_RATE_16K, SILENCE_THRESHOLD

try:
    from pydub import AudioSegment
    _HAS_PYDUB = True
except ImportError:
    _HAS_PYDUB = False

def convert_to_wav_ffmpeg(input_path: str, output_path: str) -> bool:
    if not shutil.which("ffmpeg"):
        return False
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-ar", str(SAMPLE_RATE_16K), "-ac", "1",
             "-sample_fmt", "s16", output_path],
            capture_output=True,
            timeout=30,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def convert_to_wav_pydub(input_path: str, output_path: str) -> bool:
    if not _HAS_PYDUB:
        return False
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(SAMPLE_RATE_16K).set_channels(1).set_sample_width(2)
        audio.export(output_path, format="wav")
        return os.path.exists(output_path)
    except Exception:
        return False

def resample_to_16k(input_path: str) -> str:
    out_path = input_path + "_16k.wav"
    
    if convert_to_wav_ffmpeg(input_path, out_path):
        try:
            os.remove(input_path)
        except OSError:
            pass
        return out_path

    if convert_to_wav_pydub(input_path, out_path):
        try:
            os.remove(input_path)
        except OSError:
            pass
        return out_path

    return input_path

def is_silent(frame: bytes, threshold: int = SILENCE_THRESHOLD) -> bool:
    """Return True if RMS of the PCM frame is below threshold."""
    try:
        rms = audioop.rms(frame, 2)  # 2 = sample width in bytes
        return rms < threshold
    except Exception:
        return True

def pcm8k_to_wav_bytes(pcm_frames: List[bytes]) -> bytes:
    """Pack raw 8 kHz PCM frames into an in-memory WAV."""
    import io
    buf = io.BytesIO()
    raw = b"".join(pcm_frames)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(raw)
    return buf.getvalue()
