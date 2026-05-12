from faster_whisper import WhisperModel
from typing import Optional

class WhisperAdapter:
    def __init__(self, model_size: str = "small.en", device: str = "cpu", compute_type: str = "int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model: Optional[WhisperModel] = None

    def load(self):
        if self.model is None:
            self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)

    def transcribe(self, audio_path: str):
        if self.model is None:
            self.load()
        segments, info = self.model.transcribe(audio_path, beam_size=5)
        return " ".join(s.text for s in segments).strip()
