import asyncio
import logging
from typing import Optional, Callable, Awaitable
from src.storage.conversation_buffer import ConversationBuffer
from src.storage.temp_audio_store import TempAudioStore
from src.storage.database import get_connection

log = logging.getLogger("core.post_processing")

# Module-level configuration for optional enrichment
summarise_fn: Optional[Callable[[str], Awaitable[str]]] = None
sentiment_fn: Optional[Callable[[str], Awaitable[str]]] = None

async def run_post_processing(
    session_id: str,
    caller_number: str,
    start_time: float,
    end_time: float,
    duration_s: float,
    conversation: ConversationBuffer,
    audio_store: TempAudioStore
):
    """Executes the post-call pipeline: enrichment, DB write, and cleanup."""
    log.info(f"Starting post-processing for session {session_id}")
    
    turns = conversation.get_history()
    plain_text = conversation.get_plain_text()
    
    summary = None
    sentiment = None
    
    # 1. Optional LLM Enrichment
    if summarise_fn and plain_text:
        try:
            summary = await asyncio.wait_for(summarise_fn(plain_text), timeout=30.0)
        except Exception as e:
            log.warning(f"Summarisation failed for {session_id}: {e}")

    if sentiment_fn and plain_text:
        try:
            sentiment = await asyncio.wait_for(sentiment_fn(plain_text), timeout=15.0)
        except Exception as e:
            log.warning(f"Sentiment analysis failed for {session_id}: {e}")

    # 2. Atomic Database Write
    try:
        with get_connection() as conn:
            # Record the session
            conn.execute("""
                INSERT INTO call_sessions 
                (session_id, caller_number, call_start_time, call_end_time, call_duration_s, call_status, summary, sentiment)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    call_status=excluded.call_status,
                    summary=excluded.summary,
                    sentiment=excluded.sentiment,
                    call_end_time=excluded.call_end_time,
                    call_duration_s=excluded.call_duration_s
            """, (session_id, caller_number, start_time, end_time, duration_s, "completed", summary, sentiment))

            # Record all conversation turns
            if turns:
                turn_rows = [(session_id, t.speaker, t.text, t.turn_index, t.timestamp) for t in turns]
                conn.executemany("""
                    INSERT INTO conversations (session_id, speaker, message, turn_index, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, turn_rows)
        
        # 3. Final Cleanup: Delete audio ONLY after successful DB commit
        audio_store.delete()
        log.info(f"Post-processing successfully completed for {session_id}")
        
    except Exception as e:
        log.error(f"Post-processing DB commit failed for {session_id}: {e}. Audio file preserved for recovery.")
