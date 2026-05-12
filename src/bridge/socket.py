import asyncio
import struct
import json
import logging
import httpx
import uuid
import time
from typing import List, Optional
from src.core.session import Session
from src.config.settings import GEMMA_URL, PIPER_MODEL_PATH, TTS_ENGINE
from src.config.constants import (
    KIND_UUID, KIND_AUDIO, KIND_HANGUP, FRAME_SIZE, 
    SILENCE_FRAMES_NEEDED, MIN_SPEECH_FRAMES, MAX_SPEECH_FRAMES,
    DEFAULT_SYSTEM_MSG, SILENCE_PROACTIVE_TIMEOUT, SILENCE_HANGUP_TIMEOUT
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
        self.vad = VADAdapter(threshold=0.85)
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
        caller_number = str(peer[0]) if peer else "unknown"
        session = Session(session_id=uuid.uuid4().hex, caller_number=caller_number)
        session.start()
        # 0. Fire identity fetch silently — runs while greeting plays
        asyncio.create_task(session.preload_user_context())
        
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
            
            # Silence tracking
            last_activity = time.time()
            proactive_prompted = False
            
            # Log initial greeting to session
            session.add_assistant_turn(greeting)
            history.append({"role": "assistant", "content": greeting})

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

                    session.append_audio(payload)

                    if self.vad.is_speech(payload):
                        # 🚨 BARGE-IN CHECK
                        if self.playback_task and not self.playback_task.done():
                            log.info("✂️ User interrupted AI (Speech detected)")
                            self.playback_task.cancel()
                            self.playback_task = None
                            
                        if hasattr(self, 'gemma_task') and self.gemma_task and not self.gemma_task.done():
                            log.info("✂️ User interrupted AI (Thinking)")
                            self.gemma_task.cancel()
                            self.gemma_task = None
                        
                        silence_count = 0
                        speaking = True
                        speech_buf.append(payload)
                    else:
                        silence_count += 1
                        if speaking:
                            speech_buf.append(payload)

                    # Flush speech
                    if (speaking and silence_count >= SILENCE_FRAMES_NEEDED) or len(speech_buf) >= MAX_SPEECH_FRAMES:
                        trimmed = speech_buf[:len(speech_buf)-silence_count]
                        if len(trimmed) >= MIN_SPEECH_FRAMES:
                            speech_end_time = time.time()
                            log.info(f"🎙️ Processing {len(trimmed)} frames of speech...")
                            wav_bytes = pcm8k_to_wav_bytes(trimmed)
                            
                            import tempfile
                            import os
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                                f.write(wav_bytes)
                                tmp_path = f.name
                            
                            try:
                                user_text = await asyncio.to_thread(self.whisper.transcribe, tmp_path)
                                os.remove(tmp_path)
                                t_whisper = time.time() - speech_end_time

                                if user_text:
                                    log.info(f"👤 User: {user_text} (Whisper Latency: {t_whisper:.2f}s)")
                                    session.add_user_turn(user_text)
                                    # Add user message to history once
                                    history.append({"role": "user", "content": user_text})
                                    
                                    # Wait up to 2s for context before first real LLM call.
                                    # Greeting already took ~3s to play — context is almost always ready.
                                    try:
                                        await asyncio.wait_for(session.context_ready.wait(), timeout=2.0)
                                    except asyncio.TimeoutError:
                                        log.warning("⏳ Identity resolution timed out, proceeding with current state.")

                                    # Build system prompt with whatever context is available
                                    system_prompt = self._build_system_prompt(session.user_context)

                                    # We must NOT block the read loop, otherwise Asterisk/SIP clients will drop the call due to timeout!
                                    async def process_gemma_and_speak(text_prompt, current_history, current_system_prompt, start_time):
                                        try:
                                            # Play a quick filler so the user knows we are thinking and the SIP client receives RTP (prevents 30s timeout drop)
                                            filler_pcm = await asyncio.to_thread(self.tts.synthesize_to_pcm8k, "Let me think about that...")
                                            await self.write_audio(writer, filler_pcm)
                                            
                                            gemma_start_time = time.time()
                                            reply = await self.ask_gemma(client, current_history, system_prompt=current_system_prompt, session=session)
                                            t_gemma = time.time() - gemma_start_time
                                            t_total = time.time() - start_time
                                            
                                            log.info(f"🤖 Gemma: {reply} (LLM: {t_gemma:.2f}s | Total Latency: {t_total:.2f}s)")
                                            session.add_assistant_turn(reply)
                                            
                                            # Add assistant response to history
                                            current_history.append({"role": "assistant", "content": reply})
                                            
                                            reply_pcm = await asyncio.to_thread(self.tts.synthesize_to_pcm8k, reply)
                                            # Start playback as a background task to allow interruptions
                                            self.playback_task = asyncio.create_task(self.write_audio(writer, reply_pcm))
                                        except asyncio.CancelledError:
                                            log.info("✂️ Gemma thinking task cancelled by barge-in")
                                        except Exception as e:
                                            log.error(f"❌ Transcription/LLM error: {e}")

                                    # Run asynchronously so we keep reading frames!
                                    self.gemma_task = asyncio.create_task(process_gemma_and_speak(user_text, history, system_prompt, speech_end_time))
                                    
                                else:
                                    log.info("   (empty transcription)")
                            except Exception as e:
                                log.error(f"❌ Whisper transcription error: {e}")

                        speech_buf = []
                        silence_count = 0
                        speaking = False
                        last_activity = time.time()
                        proactive_prompted = False
                    else:
                        is_thinking = hasattr(self, 'gemma_task') and self.gemma_task and not self.gemma_task.done()
                        is_speaking = self.playback_task and not self.playback_task.done()
                        
                        if is_thinking or is_speaking:
                            # Reset last_activity while the AI is busy, so we don't time out
                            last_activity = time.time()
                            silence_duration = 0
                        else:
                            # Check for silence timeouts
                            current_time = time.time()
                            silence_duration = current_time - last_activity
                        
                        if silence_duration > SILENCE_HANGUP_TIMEOUT:
                            log.info(f"⌛ Silence timeout ({SILENCE_HANGUP_TIMEOUT}s). Automatic hangup.")
                            farewell = "It seems you've gone quiet. I'll hang up for now. Feel free to call back later!"
                            farewell_pcm = await asyncio.to_thread(self.tts.synthesize_to_pcm8k, farewell)
                            await self.write_audio(writer, farewell_pcm)
                            break
                            
                        elif silence_duration > SILENCE_PROACTIVE_TIMEOUT and not proactive_prompted and not speaking:
                            log.info(f"⌛ Silence proactive timeout ({SILENCE_PROACTIVE_TIMEOUT}s). Prompting user.")
                            proactive_prompted = True
                            prompt = "Are you still there? Let me know if you need any help."
                            session.add_assistant_turn(prompt)
                            history.append({"role": "assistant", "content": prompt})
                            prompt_pcm = await asyncio.to_thread(self.tts.synthesize_to_pcm8k, prompt)
                            self.playback_task = asyncio.create_task(self.write_audio(writer, prompt_pcm))

        except Exception as e:
            log.error(f"❌ Critical error in handle_call: {e}", exc_info=True)
        finally:
            await session.end()
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

    async def ask_gemma(self, client, history, system_prompt: str = None, session: Session = None):
        """Recursively queries Gemma, executing any requested tools until a final text response is received."""
        actual_system = system_prompt or self.system_msg
        messages = [{"role": "system", "content": actual_system}] + history
        
        try:
            # Send request with tools enabled
            payload = {"messages": messages, "tools": TOOLS}
            resp = await client.post(f"{self.gemma_url}/v1/chat/completions", json=payload, timeout=120.0)
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
                        func = TOOL_MAP[name]
                        if asyncio.iscoroutinefunction(func):
                            # Special handling for identity tools: inject caller_number
                            if name == "verify_voice_token":
                                result = await func(caller_number=session.caller_number, **args)
                            elif name == "get_recent_activity":
                                result = await func(user_id=session.user_context.user_id, **args)
                            else:
                                result = await func(**args)
                        else:
                            result = func(**args)
                            
                        # Handle Mid-call authentication
                        if name == "login_to_account" and "successful" in str(result).lower():
                            from shared.identity import perform_api_login, resolve_by_token
                            username = args.get("username")
                            password = args.get("password")
                            success, token = await perform_api_login(username, password)
                            if success:
                                session.auth_token = token
                                session.user_context = await resolve_by_token(token)
                                log.info(f"🔑 Outside user successfully authenticated: {session.user_context.username}")
                        
                        # Handle Data Fetching with Token support
                        if name == "get_recent_activity" and session.auth_token:
                            # Re-run the tool with the token injected
                            result = await func(user_id=None, token=session.auth_token)

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
                return await self.ask_gemma(client, history, system_prompt=system_prompt, session=session)

            # No more tool calls, return the final text
            return message.get("content") or ""
        except Exception as e:
            log.error(f"Gemma tool loop error: {e}")
            return "Sorry, I'm having trouble responding right now."

    def _build_system_prompt(self, profile) -> str:
        """Constructs a system prompt based on Space Identity (Owner vs Guest)."""
        
        conciseness_rule = " CRITICAL INSTRUCTION: This is a spoken voice conversation. You MUST keep your responses extremely short and concise. Answer in 1 to 3 sentences maximum. DO NOT ask follow-up questions, DO NOT offer further assistance, and DO NOT say things like 'I can assist with Project Echo tasks'. Answer the prompt directly and stop talking."
        
        # GUEST / RECEPTIONIST MODE
        if profile.user_id == "":
            return (
                "You are Gemma, an AI Assistant for Project Echo. "
                "You do not recognize this local space connection. "
                "You CANNOT access any personal account data. "
                "If the caller asks for personal info, say that they need to call from their registered local space. "
                + conciseness_rule
            )

        # OWNER / PERSONAL ASSISTANT MODE
        status_str = "Active" if profile.is_active else "Inactive"
        return (
            f"You are Gemma, the private AI Assistant for {profile.username}. "
            "You have identified this call as coming from their registered Local Space. "
            f"User Profile: {profile.username}, Status: {status_str}, Joined: {profile.member_since}. "
            "You have full access to their account data. Be direct and concise. "
            + conciseness_rule
        )

        if profile.fetch_error:
            return (
                "You are Gemma, a helpful voice assistant for Project Echo. "
                f"User profile could not be loaded ({profile.fetch_error}). "
                "Help the caller as best you can without account-specific details. "
                "If they ask about their account, say the Project Echo systems are briefly unavailable."
            )

        status_str = "Active" if profile.is_active else "Inactive"
        parts = [
            "You are Gemma, a helpful voice assistant for Project Echo.",
            f"The caller is identified as {profile.username}.",
            f"Account Status: {status_str}.",
            f"Joined Project Echo on: {profile.member_since}.",
        ]
        
        if not profile.is_active:
            parts.append(
                "WARNING: this account is currently Inactive. "
                "Advise the user to contact support to reactivate their account."
            )
            
        parts.append(
            "Do not read out the user_id or email to the caller. "
            "Answer questions using the above Project Echo context."
        )
        return " ".join(parts)
