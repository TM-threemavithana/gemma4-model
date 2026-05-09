import os
import audioop
import logging
import urllib.request
import numpy as np
from typing import Optional
from src.config.settings import KOKORO_MODEL_FILE, KOKORO_VOICES_FILE, KOKORO_MODEL_DIR, KOKORO_VOICE, KOKORO_SPEED
from src.config.constants import SAMPLE_RATE_8K, SAMPLE_RATE_KOKORO

log = logging.getLogger("kokoro")

try:
    from kokoro_onnx import Kokoro
except ImportError:
    Kokoro = None

KOKORO_URLS = {
    KOKORO_MODEL_FILE:  "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    KOKORO_VOICES_FILE: "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
}

KOKORO_MIN_SIZES = {
    KOKORO_MODEL_FILE:  300 * 1024 * 1024,
    KOKORO_VOICES_FILE:  20 * 1024 * 1024,
}

class KokoroAdapter:
    def __init__(self, model_file: str = KOKORO_MODEL_FILE, voices_file: str = KOKORO_VOICES_FILE):
        self.model_file = model_file
        self.voices_file = voices_file
        self.kokoro: Optional[Kokoro] = None
        self.sample_rate = SAMPLE_RATE_KOKORO

    def _ensure_model_files(self) -> bool:
        """Validate existing files and download any that are missing or corrupt."""
        os.makedirs(KOKORO_MODEL_DIR, exist_ok=True)
        
        for file_path, url in KOKORO_URLS.items():
            if os.path.exists(file_path):
                actual_size = os.path.getsize(file_path)
                min_size = KOKORO_MIN_SIZES.get(file_path, 0)
                if actual_size >= min_size:
                    continue
                log.warning(f"  [Size Check] {os.path.basename(file_path)} looks corrupt, re-downloading...")
                os.remove(file_path)

            log.info(f"  [Download] Downloading {os.path.basename(file_path)} ...")
            tmp_path = file_path + ".part"
            try:
                urllib.request.urlretrieve(url, tmp_path)
                os.rename(tmp_path, file_path)
                log.info(f"  [OK] Saved: {os.path.basename(file_path)}")
            except Exception as e:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                log.error(f"  [Error] Download failed: {e}")
                return False
        return True

    def load(self):
        if Kokoro is None:
            raise RuntimeError("kokoro-onnx not installed. Run: pip install kokoro-onnx")
        
        if not self._ensure_model_files():
            raise RuntimeError("Could not load or download Kokoro model files.")

        log.info(f"Loading Kokoro model: {os.path.basename(self.model_file)}")
        self.kokoro = Kokoro(self.model_file, self.voices_file)
        log.info("Kokoro loaded")

    def synthesize_to_pcm8k(self, text: str, voice: str = KOKORO_VOICE) -> bytes:
        if not text or not text.strip():
            return b""
            
        if self.kokoro is None:
            self.load()
            
        samples, sample_rate = self.kokoro.create(text, voice=voice, speed=KOKORO_SPEED, lang="en-us")
        
        # Convert float32 samples to int16 PCM
        pcm_data = (samples * 32767).astype(np.int16).tobytes()
        
        # Resample from 24kHz to 8kHz for Asterisk
        resampled, _ = audioop.ratecv(
            pcm_data, 2, 1, sample_rate, SAMPLE_RATE_8K, None
        )
        return resampled
