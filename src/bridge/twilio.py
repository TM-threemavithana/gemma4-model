import asyncio
import json
import base64
import logging
import audioop
import uuid
import time
from fastapi import WebSocket, WebSocketDisconnect
from typing import List, Optional
from src.core.session import Session
from src.config.settings import GEMMA_URL, PIPER_MODEL_PATH, TTS_ENGINE
from src.config.constants import DEFAULT_SYSTEM_MSG
from src.adapters.piper import PiperAdapter
from src.adapters.kokoro import KokoroAdapter
from src.core.instances import whisper_adapter
from src.core.instances import gemma_adapter # ensure gemma is available if needed
from src.adapters.vad import VADAdapter

log = logging.getLogger("twilio_bridge")

class TwilioStreamBridge:
    def __init__(self):
        # We now use the global instances to save memory
        from src.core.instances import whisper_adapter as global_whisper
        from src.core.instances import gemma_adapter as global_gemma
        
        if TTS_ENGINE == "kokoro":
            self.tts = KokoroAdapter()
        else:
            self.tts = PiperAdapter(PIPER_MODEL_PATH)
            
        self.whisper = global_whisper
        self.vad = VADAdapter(threshold=0.5)
        self.greeting_cache: Optional[bytes] = None

    def load(self):
        self.tts.load()
        self.whisper.load()
        
    async def handle_websocket(self, websocket: WebSocket):
        await websocket.accept()
        log.info("🚀 Twilio Stream connected")
        
        stream_sid = None
        session = None
        speech_buf = []
        silence_count = 0
        last_activity = time.time()
        proactive_prompted = False
        
        try:
            while True:
                message = await websocket.receive_text()
                data = json.loads(message)
                
                if data["event"] == "start":
                    stream_sid = data["start"].get("streamSid", f"test-{uuid.uuid4().hex[:8]}")
                    caller_number = data["start"].get("from", "unknown")
                    log.info(f"📞 Call started: {stream_sid} from {caller_number}")
                    
                    session = Session(session_id=stream_sid, caller_number=caller_number)
                    session.start()
                    asyncio.create_task(session.preload_user_context())
                    
                    # Send greeting
                    greeting = "Hello! I'm Gemma, your AI assistant. How can I help you today?"
                    session.add_assistant_turn(greeting)
                    greeting_pcm = await asyncio.to_thread(self.tts.synthesize_to_pcm8k, greeting)
                    await self.send_audio(websocket, stream_sid, greeting_pcm)

                elif data["event"] == "media":
                    if not session: continue
                    
                    # Twilio sends 8000Hz mulaw
                    payload = base64.b64decode(data["media"]["payload"])
                    # Convert mulaw to linear PCM (16-bit)
                    pcm_linear = audioop.ulaw2lin(payload, 2)
                    
                    # VAD check
                    is_speech = self.vad.is_speech(pcm_linear)
                    
                    if is_speech:
                        speech_buf.append(pcm_linear)
                        silence_count = 0
                        last_activity = time.time()
                    else:
                        if speech_buf:
                            silence_count += 1
                            if silence_count > 15: # ~300ms of silence
                                log.info("🎤 Speech detected, processing...")
                                audio_bytes = b"".join(speech_buf)
                                speech_buf = []
                                silence_count = 0
                                
                                # Transcription
                                text = await asyncio.to_thread(self.whisper.transcribe, audio_bytes)
                                if text.strip():
                                    log.info(f"👤 User: {text}")
                                    session.add_user_turn(text)
                                    
                                    # Gemma Response
                                    response_text = await session.generate_response()
                                    log.info(f"🤖 Gemma: {response_text}")
                                    
                                    # TTS
                                    resp_pcm = await asyncio.to_thread(self.tts.synthesize_to_pcm8k, response_text)
                                    await self.send_audio(websocket, stream_sid, resp_pcm)

                elif data["event"] == "stop":
                    log.info(f"🛑 Stream stopped: {stream_sid}")
                    break
                    
        except WebSocketDisconnect:
            log.info("🔌 WebSocket disconnected")
        except Exception as e:
            log.error(f"❌ Bridge error: {e}", exc_info=True)
        finally:
            if session:
                session.end()

    async def send_audio(self, websocket, stream_sid, pcm_data):
        """Sends PCM linear 8k audio to Twilio as mulaw base64."""
        # Convert linear to mulaw
        mulaw_data = audioop.lin2ulaw(pcm_data, 2)
        
        # Twilio wants small chunks (~20ms = 160 bytes of mulaw)
        chunk_size = 160
        for i in range(0, len(mulaw_data), chunk_size):
            chunk = mulaw_data[i:i+chunk_size]
            message = {
                "event": "media",
                "streamSid": stream_sid,
                "media": {
                    "payload": base64.b64encode(chunk).decode("utf-8")
                }
            }
            await websocket.send_text(json.dumps(message))
