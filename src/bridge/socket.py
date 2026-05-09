import asyncio
import struct
import json
import logging
import httpx
from typing import List, Optional
from src.config.settings import GEMMA_URL, PIPER_MODEL_PATH, TTS_ENGINE
from src.config.constants import (
    KIND_UUID, KIND_AUDIO, KIND_HANGUP, FRAME_SIZE, 
    SILENCE_FRAMES_NEEDED, MIN_SPEECH_FRAMES, MAX_SPEECH_FRAMES,
    DEFAULT_SYSTEM_MSG
)
from src.adapters.piper import PiperAdapter
from src.adapters.kokoro import KokoroAdapter
from src.adapters.whisper import WhisperAdapter
from src.adapters.vad import VADAdapter
from src.core.audio import pcm8k_to_wav_bytes
from src.tools.registry import TOOLS, TOOL_MAP

log = logging.getLogger("bridge")

class AudioSocketBridge:
    def __init__(self, gemma_url: str = GEMMA_URL, system_msg: str = DEFAULT_SYSTEM_MSG):
        self.gemma_url = gemma_url
        self.system_msg = system_msg
        
        if TTS_ENGINE == "kokoro":
            self.tts = KokoroAdapter()
        else:
            self.tts = PiperAdapter(PIPER_MODEL_PATH)
            
        self.whisper = WhisperAdapter()
        self.vad = VADAdapter(threshold=0.5)
        self.greeting_cache: Optional[bytes] = None
        self.playback_task: Optional[asyncio.Task] = None

    async def start(self, port: int):
        self.tts.load()
        self.whisper.load()
        
        # Pre-synthesize greeting to eliminate delay
        greeting = "Hello! I'm Gemma, your AI assistant. How can I help you today?"
        log.info("🔊 Pre-synthesizing greeting...")
        try:
            self.greeting_cache = await asyncio.to_thread(self.tts.synthesize_to_pcm8k, greeting)
            log.info("✅ Greeting ready")
        except Exception as e:
            log.error(f"❌ Pre-synthesis failed: {e}")

        server = await asyncio.start_server(self.handle_call, "0.0.0.0", port)
        log.info(f"✅ AudioSocket bridge listening on port {port}")
        async with server:
            await server.serve_forever()

    async def handle_call(self, reader, writer):
        peer = writer.get_extra_info("peername")
        log.info(f"📞 Incoming call from {peer}")
        
        try:
            # 1. Read UUID frame (first frame Asterisk always sends)
            kind, payload = await self.read_frame(reader)
            if kind == KIND_UUID:
                call_uuid = payload.hex() if payload else "unknown"
                log.info(f"   [Handshake] Call UUID: {call_uuid}")
            elif kind is not None:
                log.warning(f"   [Handshake] First frame was not UUID (kind=0x{kind:02x})")

            # 2. Start initial greeting as a background task (interruptible)
            greeting = "Hello! I'm Gemma, your AI assistant. How can I help you today?"
            log.info(f"🔊 Starting greeting...")
            try:
                if self.greeting_cache:
                    greeting_pcm = self.greeting_cache
                else:
                    greeting_pcm = await asyncio.to_thread(self.tts.synthesize_to_pcm8k, greeting)
                self.playback_task = asyncio.create_task(self.write_audio(writer, greeting_pcm))
            except Exception as e:
                log.error(f"❌ Greeting setup failed: {e}")

            # 3. Conversation loop
            history = []
            speech_buf = []
            silence_count = 0
            speaking = False

            async with httpx.AsyncClient() as client:
                while True:
                    kind, payload = await self.read_frame(reader)
                    
                    if kind is None:
                        log.info(f"📵 Connection closed by peer ({peer})")
                        break
                    
                    if kind == KIND_HANGUP:
                        log.info(f"📵 Hangup received from Asterisk ({peer})")
                        break
                    
                    if kind != KIND_AUDIO:
                        log.debug(f"   Skipping non-audio frame (kind=0x{kind:02x})")
                        continue

                    if self.vad.is_speech(payload):
                        # 🚨 BARGE-IN CHECK
                        if self.playback_task and not self.playback_task.done():
                            log.info("✂️ User interrupted AI (Speech detected)")
                            self.playback_task.cancel()
                            self.playback_task = None
                        
                        silence_count = 0
                        speaking = True
                        speech_buf.append(payload)
                    else:
                        silence_count += 1
                        if speaking:
                            speech_buf.append(payload)

                    # Flush speech
                    if (speaking and silence_count >= SILENCE_FRAMES_NEEDED) or len(speech_buf) >= MAX_SPEECH_FRAMES:
                        if len(speech_buf) >= MIN_SPEECH_FRAMES:
                            log.info(f"🎙️ Processing {len(speech_buf)} frames of speech...")
                            trimmed = speech_buf[:len(speech_buf)-silence_count]
                            wav_bytes = pcm8k_to_wav_bytes(trimmed)
                            
                            import tempfile
                            import os
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                                f.write(wav_bytes)
                                tmp_path = f.name
                            
                            try:
                                user_text = await asyncio.to_thread(self.whisper.transcribe, tmp_path)
                                os.remove(tmp_path)

                                if user_text:
                                    log.info(f"👤 User: {user_text}")
                                    # Add user message to history once
                                    history.append({"role": "user", "content": user_text})
                                    
                                    # Ask Gemma (this will handle tools internally)
                                    reply = await self.ask_gemma(client, history)
                                    log.info(f"🤖 Gemma: {reply}")
                                    
                                    # Add assistant response to history
                                    history.append({"role": "assistant", "content": reply})
                                    
                                    reply_pcm = await asyncio.to_thread(self.tts.synthesize_to_pcm8k, reply)
                                    # Start playback as a background task to allow interruptions
                                    self.playback_task = asyncio.create_task(self.write_audio(writer, reply_pcm))
                                else:
                                    log.info("   (empty transcription)")
                            except Exception as e:
                                log.error(f"❌ Transcription/LLM error: {e}")

                        speech_buf = []
                        silence_count = 0
                        speaking = False

        except Exception as e:
            log.error(f"❌ Critical error in handle_call: {e}", exc_info=True)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except:
                pass
            log.info(f"📵 Call session ended ({peer})")

    async def read_frame(self, reader):
        """Read one AudioSocket frame: [1-byte kind][2-byte length][payload]"""
        try:
            header = await reader.readexactly(3)
            kind = header[0]
            length = struct.unpack(">H", header[1:3])[0]
            payload = await reader.readexactly(length) if length > 0 else b""
            return kind, payload
        except asyncio.IncompleteReadError:
            return None, None
        except Exception as e:
            log.debug(f"read_frame error: {e}")
            return None, None

    async def write_audio(self, writer, pcm_data):
        """Sends audio to Asterisk in real-time chunks (20ms) to allow for interruption."""
        try:
            offset = 0
            while offset < len(pcm_data):
                chunk = pcm_data[offset:offset+FRAME_SIZE]
                if len(chunk) < FRAME_SIZE:
                    chunk += b"\x00" * (FRAME_SIZE - len(chunk))
                
                frame = bytes([KIND_AUDIO]) + struct.pack(">H", FRAME_SIZE) + chunk
                writer.write(frame)
                await writer.drain()
                
                offset += FRAME_SIZE
                # FRAME_SIZE is 320 bytes (160 samples at 8kHz) = 20ms of audio
                await asyncio.sleep(0.02)
        except asyncio.CancelledError:
            # Task was cancelled due to barge-in
            raise
        except Exception as e:
            log.error(f"Error in write_audio: {e}")

    async def ask_gemma(self, client, history):
        """Recursively queries Gemma, executing any requested tools until a final text response is received."""
        messages = [{"role": "system", "content": self.system_msg}] + history
        
        try:
            # Send request with tools enabled
            payload = {"messages": messages, "tools": TOOLS}
            resp = await client.post(f"{self.gemma_url}/v1/chat/completions", json=payload, timeout=30)
            data = resp.json()
            message = data["choices"][0]["message"]
            
            # If Gemma wants to call a tool
            if "tool_calls" in message:
                tool_calls = message["tool_calls"]
                # Add the tool call message to history
                history.append(message)
                
                for tool_call in tool_calls:
                    func_info = tool_call.get("function", {})
                    name = func_info.get("name")
                    args_raw = func_info.get("arguments", "{}")
                    
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    
                    if name in TOOL_MAP:
                        log.info(f"🛠️ Executing tool: {name}({args})")
                        result = TOOL_MAP[name](**args)
                        
                        # Add tool result to history
                        history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.get("id", "call_123"),
                            "name": name,
                            "content": str(result)
                        })
                    else:
                        log.warning(f"⚠️ Tool {name} not found in registry")

                # Re-query Gemma with the tool results included in history
                return await self.ask_gemma(client, history)

            # No more tool calls, return the final text
            return message.get("content") or ""
        except Exception as e:
            log.error(f"Gemma tool loop error: {e}")
            return "Sorry, I'm having trouble responding right now."
