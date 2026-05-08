import asyncio
import threading
from typing import Optional, List, Dict, Any
from src.adapters.gemma import GemmaAdapter
from src.adapters.whisper import WhisperAdapter
from src.config.settings import TEMPERATURE, MAX_TOKENS

class GemmaAgent:
    def __init__(self, gemma: GemmaAdapter, whisper: Optional[WhisperAdapter] = None):
        self.gemma = gemma
        self.whisper = whisper
        self._audio_lock = threading.Lock()

    async def generate_text(self, prompt: str, history: Optional[List[Dict[str, str]]] = None, system_msg: Optional[str] = None) -> str:
        conv_kwargs = {}
        if system_msg:
            conv_kwargs["system_message"] = system_msg
        if history:
            conv_kwargs["messages"] = history

        last_message = {"role": "user", "content": [{"type": "text", "text": prompt}]}
        
        return await asyncio.to_thread(self._sync_generate, last_message, conv_kwargs)

    def _sync_generate(self, last_message, conv_kwargs):
        gen_kwargs = {"max_decode_steps": MAX_TOKENS, "temperature": TEMPERATURE}
        with self.gemma.create_conversation(**conv_kwargs) as conversation:
            try:
                response = conversation.send_message(last_message, **gen_kwargs)
            except TypeError:
                response = conversation.send_message(last_message)
            
            if isinstance(response, dict):
                parts = response.get("content", [])
                return "".join(p.get("text", "") for p in parts if p.get("type") == "text")
            return str(response)

    async def generate_from_audio(self, audio_path: str, prompt: str) -> str:
        if self.whisper:
            transcript = await asyncio.to_thread(self.whisper.transcribe, audio_path)
            if transcript:
                full_prompt = f"{prompt}\n\nAudio content: {transcript}"
                return await self.generate_text(full_prompt)
            else:
                return "Audio was silent or could not be transcribed."
        
        # Fallback to native audio inference if no whisper adapter
        with self._audio_lock:
            user_message = {
                "role": "user",
                "content": [
                    {"type": "audio", "path": audio_path},
                    {"type": "text", "text": prompt},
                ],
            }
            return await asyncio.to_thread(self._sync_generate, user_message, {})
