import os
import audioop
from typing import Optional
from src.config.settings import PIPER_MODEL_PATH
from src.config.constants import SAMPLE_RATE_8K

try:
    from piper.voice import PiperVoice
except ImportError:
    PiperVoice = None

class PiperAdapter:
    def __init__(self, model_path: str = PIPER_MODEL_PATH):
        self.model_path = model_path
        self.voice: Optional[PiperVoice] = None
        self.sample_rate: int = 22050

    def load(self):
        if PiperVoice is None:
            raise RuntimeError("piper-tts is not installed.")
        
        config_path = self.model_path + ".json"
        if not os.path.exists(self.model_path) or not os.path.exists(config_path):
            raise FileNotFoundError(f"Piper model or config not found at {self.model_path}")
            
        self.voice = PiperVoice.load(self.model_path, config_path=config_path)
        self.sample_rate = self.voice.config.sample_rate

    def synthesize_to_pcm8k(self, text: str) -> bytes:
        if self.voice is None:
            self.load()
            
        raw_audio = b"".join(chunk.audio_int16_bytes for chunk in self.voice.synthesize(text))
        # Resample to 8kHz for Asterisk
        resampled, _ = audioop.ratecv(
            raw_audio, 2, 1, self.sample_rate, SAMPLE_RATE_8K, None
        )
        return resampled
