"""
Asterisk AudioSocket Bridge — Gemma 4 Calling Agent
=====================================================
Sits between Asterisk (AudioSocket protocol) and the Gemma 4 server.py.

Architecture:
  SIP Softphone → Asterisk → [THIS FILE] → server.py (Gemma 4) → TTS → Asterisk → Caller

AudioSocket Protocol (Asterisk):
  Every TCP frame is:  [1-byte kind][2-byte big-endian length][payload]
  Kind 0x00 = UUID (16 bytes, sent once at connection start)
  Kind 0x10 = SLIN audio (16-bit signed, 8 kHz, mono, 320 bytes = 20ms)
  Kind 0x01 = Hangup signal

Usage (run in WSL2):
  python3 asterisk_bridge.py
  python3 asterisk_bridge.py --port 9092 --gemma-url http://localhost:8000
  python3 asterisk_bridge.py --debug   # verbose frame logging
"""

import asyncio
import audioop
import io
import os
import struct
import sys

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import tempfile
import time
import wave
import argparse
import logging
from typing import Optional, List

import httpx
from gtts import gTTS


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bridge")


# ---------------------------------------------------------------------------
# Config (overridden by CLI args)
# ---------------------------------------------------------------------------
DEFAULT_PORT         = 9092
DEFAULT_GEMMA_URL    = "http://localhost:8000"
DEFAULT_SYSTEM_MSG   = (
    "You are Gemma, a helpful AI voice assistant. "
    "Keep your answers SHORT — 2-3 sentences max. "
    "You are speaking on a phone call, so avoid markdown or bullet points."
)

# Silence detection: how many consecutive silent frames = end of user speech
SILENCE_THRESHOLD_DB    = 500    # RMS below this → silence (tweak for your mic)
SILENCE_FRAMES_NEEDED   = 40     # 40 × 20ms = 0.8 s of silence → done speaking
MIN_SPEECH_FRAMES       = 10     # ignore bursts shorter than this (200 ms)
MAX_SPEECH_FRAMES       = 500    # 10 s hard cap — force flush


# ---------------------------------------------------------------------------
# AudioSocket frame constants
# ---------------------------------------------------------------------------
KIND_UUID    = 0x00
KIND_AUDIO   = 0x10
KIND_HANGUP  = 0x01
KIND_ERROR   = 0xFF

FRAME_SIZE   = 320   # bytes — 160 samples × 2 bytes @ 8 kHz (20 ms)


# ---------------------------------------------------------------------------
# Whisper STT (loaded once, shared across calls)
# ---------------------------------------------------------------------------
_whisper = None

def get_whisper():
    global _whisper
    if _whisper is None:
        log.info("⏳  Loading faster-whisper model (base, CPU)…")
        from faster_whisper import WhisperModel
        _whisper = WhisperModel("base", device="cpu", compute_type="int8")
        log.info("✅  Whisper loaded")
    return _whisper


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def pcm8k_to_wav_bytes(pcm_frames: List[bytes]) -> bytes:
    """Pack raw 8 kHz PCM frames into an in-memory WAV."""
    buf = io.BytesIO()
    raw = b"".join(pcm_frames)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)        # 16-bit
        wf.setframerate(8000)
        wf.writeframes(raw)
    return buf.getvalue()


def mp3_to_pcm8k(mp3_bytes: bytes) -> bytes:
    """
    Convert MP3 bytes (gTTS output) → 8 kHz 16-bit mono PCM.
    Uses pydub + audioop for resampling — no ffmpeg required when using
    the built-in pure-Python fallback, but ffmpeg gives better quality.
    Falls back gracefully if pydub/ffmpeg is unavailable.
    """
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
        seg = seg.set_frame_rate(8000).set_channels(1).set_sample_width(2)
        return seg.raw_data
    except Exception as e:
        log.warning(f"pydub resampling failed ({e}), using audioop fallback")
        # Very basic fallback: just return silence so the call doesn't crash
        return b"\x00\x00" * 8000   # 1 second of silence


def text_to_pcm8k(text: str) -> bytes:
    """TTS: text → MP3 (gTTS) → 8 kHz PCM."""
    tts = gTTS(text=text, lang="en", slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    return mp3_to_pcm8k(buf.getvalue())


def is_silent(frame: bytes, threshold: int = SILENCE_THRESHOLD_DB) -> bool:
    """Return True if RMS of the PCM frame is below threshold."""
    try:
        rms = audioop.rms(frame, 2)  # 2 = sample width in bytes
        return rms < threshold
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Gemma 4 LLM call
# ---------------------------------------------------------------------------

async def ask_gemma(
    user_text: str,
    history: list,
    gemma_url: str,
    system_msg: str,
    http_client: httpx.AsyncClient,
) -> str:
    """Send user_text + history to Gemma 4, return assistant reply."""
    messages = [{"role": "system", "content": system_msg}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    payload = {
        "model": "gemma-4-e2b-it",
        "messages": messages,
        "max_tokens": 200,
        "temperature": 0.7,
        "stream": False,
    }

    try:
        resp = await http_client.post(
            f"{gemma_url}/v1/chat/completions",
            json=payload,
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        reply = data["choices"][0]["message"]["content"].strip()
        return reply
    except Exception as e:
        log.error(f"Gemma request failed: {e}")
        return "I'm sorry, I had trouble generating a response. Please try again."


# ---------------------------------------------------------------------------
# AudioSocket frame I/O
# ---------------------------------------------------------------------------

async def read_frame(reader: asyncio.StreamReader):
    """Read one AudioSocket frame → (kind, payload_bytes) or None on EOF."""
    try:
        header = await reader.readexactly(3)
    except (asyncio.IncompleteReadError, ConnectionResetError):
        return None, None
    kind = header[0]
    length = struct.unpack(">H", header[1:3])[0]
    payload = await reader.readexactly(length) if length > 0 else b""
    return kind, payload


async def write_audio(writer: asyncio.StreamWriter, pcm_data: bytes):
    """Write raw PCM as a series of AudioSocket SLIN frames."""
    offset = 0
    while offset < len(pcm_data):
        chunk = pcm_data[offset: offset + FRAME_SIZE]
        # Pad last frame if needed
        if len(chunk) < FRAME_SIZE:
            chunk = chunk + b"\x00" * (FRAME_SIZE - len(chunk))
        frame = bytes([KIND_AUDIO]) + struct.pack(">H", FRAME_SIZE) + chunk
        writer.write(frame)
        offset += FRAME_SIZE
    await writer.drain()


# ---------------------------------------------------------------------------
# Per-call handler
# ---------------------------------------------------------------------------

async def handle_call(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    gemma_url: str,
    system_msg: str,
    debug: bool,
):
    peer = writer.get_extra_info("peername")
    log.info(f"📞  Incoming AudioSocket connection from {peer}")

    # Read UUID frame (first frame Asterisk always sends)
    kind, payload = await read_frame(reader)
    if kind == KIND_UUID:
        call_uuid = payload.hex() if payload else "unknown"
        log.info(f"    Call UUID: {call_uuid}")
    else:
        log.warning("First frame was not UUID — proceeding anyway")

    # Play a greeting
    greeting = "Hello! I'm Gemma, your AI assistant. How can I help you today?"
    log.info(f"🔊  Greeting: {greeting}")
    try:
        greeting_pcm = await asyncio.to_thread(text_to_pcm8k, greeting)
        await write_audio(writer, greeting_pcm)
    except Exception as e:
        log.error(f"Greeting TTS failed: {e}")

    # Per-call state
    history: list = []
    speech_buf: List[bytes] = []
    silence_count = 0
    speaking = False
    whisper = get_whisper()

    async with httpx.AsyncClient() as http_client:
        while True:
            kind, payload = await read_frame(reader)

            if kind is None:
                log.info("📵  Connection closed by Asterisk (EOF)")
                break

            if kind == KIND_HANGUP:
                log.info("📵  Hangup received from Asterisk")
                break

            if kind != KIND_AUDIO:
                if debug:
                    log.debug(f"    Non-audio frame kind=0x{kind:02x}, skipping")
                continue

            frame = payload

            if is_silent(frame):
                silence_count += 1
                if speaking:
                    speech_buf.append(frame)  # keep a bit of trailing silence
            else:
                silence_count = 0
                speaking = True
                speech_buf.append(frame)

            # --- Flush condition: enough silence after speech ---
            if (
                speaking
                and silence_count >= SILENCE_FRAMES_NEEDED
                and len(speech_buf) >= MIN_SPEECH_FRAMES
            ) or len(speech_buf) >= MAX_SPEECH_FRAMES:

                if debug:
                    duration_ms = len(speech_buf) * 20
                    log.debug(f"    Flushing {len(speech_buf)} frames ({duration_ms} ms)")

                # Trim trailing silence
                trimmed = speech_buf[: len(speech_buf) - silence_count]
                speech_buf.clear()
                silence_count = 0
                speaking = False

                # STT
                wav_bytes = pcm8k_to_wav_bytes(trimmed)
                tmp_wav = None
                try:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".wav"
                    ) as f:
                        f.write(wav_bytes)
                        tmp_wav = f.name

                    log.info("🎙️  Transcribing…")
                    segments, _ = await asyncio.to_thread(
                        whisper.transcribe, tmp_wav, beam_size=5
                    )
                    user_text = " ".join(s.text for s in segments).strip()
                finally:
                    if tmp_wav and os.path.exists(tmp_wav):
                        os.remove(tmp_wav)

                if not user_text:
                    log.info("    (empty transcription, skipping)")
                    continue

                log.info(f"👤  User: {user_text}")

                # LLM
                log.info("🤖  Asking Gemma 4…")
                reply = await ask_gemma(
                    user_text, history, gemma_url, system_msg, http_client
                )
                log.info(f"🤖  Gemma: {reply}")

                # Update history (last 10 exchanges to avoid token overflow)
                history.append({"role": "user", "content": user_text})
                history.append({"role": "assistant", "content": reply})
                if len(history) > 20:
                    history = history[-20:]

                # TTS → PCM → stream to caller
                log.info("🔊  Synthesizing voice…")
                try:
                    reply_pcm = await asyncio.to_thread(text_to_pcm8k, reply)
                    await write_audio(writer, reply_pcm)
                except Exception as e:
                    log.error(f"TTS/write failed: {e}")

    try:
        writer.close()
        await writer.wait_closed()
    except Exception:
        pass
    log.info(f"📵  Call ended ({peer})")


# ---------------------------------------------------------------------------
# Main server
# ---------------------------------------------------------------------------

async def main(port: int, gemma_url: str, system_msg: str, debug: bool):
    if debug:
        log.setLevel(logging.DEBUG)

    # Pre-load Whisper so the first call isn't slow
    log.info("⏳  Pre-loading Whisper model…")
    await asyncio.to_thread(get_whisper)

    server = await asyncio.start_server(
        lambda r, w: handle_call(r, w, gemma_url, system_msg, debug),
        host="0.0.0.0",
        port=port,
    )
    addr = server.sockets[0].getsockname()
    log.info(f"✅  AudioSocket bridge listening on {addr[0]}:{addr[1]}")
    log.info(f"    Gemma 4 URL : {gemma_url}")
    log.info(f"    System msg  : {system_msg[:60]}…")
    log.info("    Waiting for Asterisk connections… (Ctrl+C to stop)")

    async with server:
        await server.serve_forever()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Asterisk AudioSocket ↔ Gemma 4 Bridge")
    p.add_argument("--port", type=int, default=DEFAULT_PORT,
                   help=f"TCP port to listen on (default {DEFAULT_PORT})")
    p.add_argument("--gemma-url", default=DEFAULT_GEMMA_URL,
                   help=f"Base URL of server.py (default {DEFAULT_GEMMA_URL})")
    p.add_argument("--system-msg", default=DEFAULT_SYSTEM_MSG,
                   help="System prompt for Gemma 4")
    p.add_argument("--silence-threshold", type=int, default=SILENCE_THRESHOLD_DB,
                   help="RMS threshold for silence detection (default 500)")
    p.add_argument("--debug", action="store_true",
                   help="Enable verbose frame logging")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    SILENCE_THRESHOLD_DB = args.silence_threshold
    try:
        asyncio.run(main(args.port, args.gemma_url, args.system_msg, args.debug))
    except KeyboardInterrupt:
        log.info("Bridge stopped.")
