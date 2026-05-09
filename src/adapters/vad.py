import logging
import numpy as np
from typing import List
try:
    from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
except ImportError:
    load_silero_vad = None

log = logging.getLogger("vad")

class VADAdapter:
    def __init__(self, threshold: float = 0.5):
        """
        threshold: 0.0-1.0 (Higher is more aggressive at filtering noise)
        """
        if load_silero_vad is None:
            raise RuntimeError("silero-vad not installed. Run: pip install silero-vad")
        
        self.model = load_silero_vad()
        self.threshold = threshold
        self.sample_rate = 8000
        
    def is_speech(self, pcm_frame: bytes) -> bool:
        """
        Analyzes a PCM frame (8kHz mono).
        """
        try:
            # Convert bytes to float32 numpy array as expected by Silero
            audio_float32 = np.frombuffer(pcm_frame, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Use get_speech_timestamps or the raw model call
            # For real-time, we can just check the probability
            import torch
            audio_tensor = torch.from_numpy(audio_float32)
            
            speech_prob = self.model(audio_tensor, self.sample_rate).item()
            return speech_prob > self.threshold
        except Exception as e:
            log.debug(f"Silero VAD Error: {e}")
            return False

    def contains_speech(self, pcm_data: bytes) -> bool:
        """Helper for larger buffers."""
        audio_float32 = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
        import torch
        audio_tensor = torch.from_numpy(audio_float32)
        speech_prob = self.model(audio_tensor, self.sample_rate).item()
        return speech_prob > self.threshold
