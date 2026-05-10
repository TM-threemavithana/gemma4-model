import asyncio
import time
import logging
from src.storage.conversation_buffer import ConversationBuffer
from src.storage.temp_audio_store import TempAudioStore
from src.storage.database import init_db
from src.core.post_processing import run_post_processing

log = logging.getLogger("core.session")

class Session:
    def __init__(self, session_id: str, caller_number: str):
        self.session_id = session_id
        self.caller_number = caller_number
        self.start_time = time.time()
        self.conversation = ConversationBuffer(session_id, caller_number)
        self.audio = TempAudioStore(session_id)

    def start(self):
        """Initializes storage and opens resources."""
        init_db()
        self.audio.open()
        log.info(f"Session {self.session_id} started for {self.caller_number}")

    def add_user_turn(self, text: str):
        """Records user transcription."""
        self.conversation.add_user_turn(text)

    def add_assistant_turn(self, text: str):
        """Records assistant response."""
        self.conversation.add_assistant_turn(text)

    def append_audio(self, pcm_bytes: bytes):
        """Streams incoming audio to temporary storage."""
        self.audio.append_chunk(pcm_bytes)

    async def end(self):
        """Closes resources and launches asynchronous post-processing."""
        self.audio.close()
        end_time = time.time()
        duration_s = end_time - self.start_time
        
        log.info(f"Session {self.session_id} ending (duration: {duration_s:.1f}s). Firing post-processing.")
        
        # Launch fire-and-forget background task
        asyncio.ensure_future(run_post_processing(
            session_id=self.session_id,
            caller_number=self.caller_number,
            start_time=self.start_time,
            end_time=end_time,
            duration_s=duration_s,
            conversation=self.conversation,
            audio_store=self.audio
        ))
