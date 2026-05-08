import asyncio
import struct
import logging
import httpx
from typing import List, Optional
from src.config.settings import GEMMA_URL, PIPER_MODEL_PATH
from src.config.constants import (
    KIND_UUID, KIND_AUDIO, KIND_HANGUP, FRAME_SIZE, 
    SILENCE_FRAMES_NEEDED, MIN_SPEECH_FRAMES, MAX_SPEECH_FRAMES,
    DEFAULT_SYSTEM_MSG
)
from src.adapters.piper import PiperAdapter
from src.adapters.whisper import WhisperAdapter
from src.core.audio import is_silent, pcm8k_to_wav_bytes

log = logging.getLogger("bridge")

class AudioSocketBridge:
    def __init__(self, gemma_url: str = GEMMA_URL, system_msg: str = DEFAULT_SYSTEM_MSG):
        self.gemma_url = gemma_url
        self.system_msg = system_msg
        self.piper = PiperAdapter(PIPER_MODEL_PATH)
        self.whisper = WhisperAdapter()

    async def start(self, port: int):
        self.piper.load()
        self.whisper.load()
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

            # 2. Play initial greeting
            greeting = "Hello! I'm Gemma, your AI assistant. How can I help you today?"
            log.info(f"🔊 Greeting: {greeting}")
            try:
                greeting_pcm = await asyncio.to_thread(self.piper.synthesize_to_pcm8k, greeting)
                await self.write_audio(writer, greeting_pcm)
            except Exception as e:
                log.error(f"❌ Greeting synthesis failed: {e}")

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

                    if is_silent(payload):
                        silence_count += 1
                        if speaking:
                            speech_buf.append(payload)
                    else:
                        silence_count = 0
                        speaking = True
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
                                    reply = await self.ask_gemma(client, user_text, history)
                                    log.info(f"🤖 Gemma: {reply}")
                                    
                                    history.append({"role": "user", "content": user_text})
                                    history.append({"role": "assistant", "content": reply})
                                    
                                    reply_pcm = await asyncio.to_thread(self.piper.synthesize_to_pcm8k, reply)
                                    await self.write_audio(writer, reply_pcm)
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
        offset = 0
        while offset < len(pcm_data):
            chunk = pcm_data[offset:offset+FRAME_SIZE]
            if len(chunk) < FRAME_SIZE:
                chunk += b"\x00" * (FRAME_SIZE - len(chunk))
            frame = bytes([KIND_AUDIO]) + struct.pack(">H", FRAME_SIZE) + chunk
            writer.write(frame)
            offset += FRAME_SIZE
        await writer.drain()

    async def ask_gemma(self, client, text, history):
        messages = [{"role": "system", "content": self.system_msg}] + history + [{"role": "user", "content": text}]
        try:
            resp = await client.post(f"{self.gemma_url}/v1/chat/completions", json={"messages": messages}, timeout=30)
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            log.error(f"Gemma error: {e}")
            return "Sorry, I had trouble thinking."
