import os
import wave
import logging
from src.config.settings import TEMP_AUDIO_DIR

log = logging.getLogger("storage.temp_audio")

class TempAudioStore:
    def __init__(self, session_id: str):
        self.session_id = session_id
        os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
        self.file_path = os.path.join(TEMP_AUDIO_DIR, f"{session_id}.wav")
        self._wav_file = None
        self._is_open = False

    def open(self):
        try:
            self._wav_file = wave.open(self.file_path, 'wb')
            self._wav_file.setnchannels(1)
            self._wav_file.setsampwidth(2) # 16-bit PCM
            self._wav_file.setframerate(8000) # 8kHz
            self._is_open = True
            log.debug(f"Opened temp audio file: {self.file_path}")
        except Exception as e:
            log.error(f"Failed to open temp audio file: {e}")

    def append_chunk(self, pcm_bytes: bytes):
        """Appends a raw PCM chunk to the WAV file."""
        if self._is_open and self._wav_file:
            try:
                self._wav_file.writeframes(pcm_bytes)
            except Exception as e:
                log.error(f"Failed to write audio chunk: {e}")

    def close(self):
        if self._is_open and self._wav_file:
            try:
                self._wav_file.close()
                self._is_open = False
                log.debug(f"Closed temp audio file: {self.file_path}")
            except Exception as e:
                log.error(f"Failed to close temp audio file: {e}")

    def delete(self):
        """Closes and deletes the temporary audio file."""
        self.close()
        try:
            if os.path.exists(self.file_path):
                os.remove(self.file_path)
                log.debug(f"Deleted temp audio file: {self.file_path}")
        except Exception as e:
            log.error(f"Failed to delete temp audio file: {e}")
