"""
test_bridge.py — Simulate an Asterisk AudioSocket connection
============================================================
Tests the full pipeline (STT → Gemma 4 → TTS) WITHOUT needing Asterisk or a
phone.  It pretends to be Asterisk and sends a WAV file as raw PCM audio to
the bridge, then captures and saves the audio response.

Requirements:
  1. asterisk_bridge.py must be running: python3 asterisk_bridge.py
  2. server.py (Gemma 4) must be running: python server.py

Usage:
  # Test with a WAV file:
  python test_bridge.py --wav path/to/your_voice.wav

  # Test with a hardcoded text (auto-generates speech via gTTS):
  python test_bridge.py --text "Hello Gemma, what is 2 plus 2?"

  # Quick smoke test (sends a synthetic beep):
  python test_bridge.py --smoke
"""

import asyncio
import io
import os
import struct
import sys

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import wave
import argparse
import time
import uuid

# ---------------------------------------------------------------------------
# AudioSocket protocol constants (mirrors asterisk_bridge.py)
# ---------------------------------------------------------------------------
KIND_UUID   = 0x00
KIND_AUDIO  = 0x10
KIND_HANGUP = 0x01

FRAME_SIZE = 320    # 20 ms of 8 kHz 16-bit mono


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_frame(kind: int, payload: bytes) -> bytes:
    return bytes([kind]) + struct.pack(">H", len(payload)) + payload


def wav_to_pcm8k(wav_path: str) -> bytes:
    """Read a WAV file and resample/convert to 8 kHz 16-bit mono PCM."""
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(wav_path)
        seg = seg.set_frame_rate(8000).set_channels(1).set_sample_width(2)
        return seg.raw_data
    except Exception:
        # Fallback: just read raw bytes if pydub unavailable
        with wave.open(wav_path, "rb") as wf:
            if wf.getframerate() != 8000 or wf.getnchannels() != 1:
                print("⚠️  WAV is not 8 kHz mono — pydub unavailable, may sound wrong")
            return wf.readframes(wf.getnframes())


def text_to_pcm8k(text: str) -> bytes:
    """Convert text to speech using gTTS, then resample to 8 kHz PCM."""
    from gtts import gTTS
    from pydub import AudioSegment
    buf = io.BytesIO()
    gTTS(text=text, lang="en").write_to_fp(buf)
    buf.seek(0)
    seg = AudioSegment.from_mp3(buf)
    seg = seg.set_frame_rate(8000).set_channels(1).set_sample_width(2)
    return seg.raw_data


def generate_beep_pcm(duration_ms: int = 1000, freq_hz: int = 440) -> bytes:
    """Generate a pure sine-wave beep as 8 kHz 16-bit mono PCM."""
    import math
    n_samples = int(8000 * duration_ms / 1000)
    samples = []
    for i in range(n_samples):
        val = int(16000 * math.sin(2 * math.pi * freq_hz * i / 8000))
        samples.append(struct.pack("<h", val))
    return b"".join(samples)


def pcm_to_wav(pcm: bytes, output_path: str):
    """Save raw 8 kHz PCM as a WAV file so you can listen to it."""
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(pcm)


async def run_test(
    host: str,
    port: int,
    pcm_input: bytes,
    output_wav: str,
    listen_seconds: float = 8.0,
):
    """
    Connect to the bridge, send UUID + audio frames, capture audio response.
    """
    print(f"\n🔌  Connecting to bridge at {host}:{port}…")
    reader, writer = await asyncio.open_connection(host, port)
    print("✅  Connected")

    # --- Send fake UUID frame (16 random bytes) ---
    fake_uuid = uuid.uuid4().bytes
    writer.write(build_frame(KIND_UUID, fake_uuid))
    await writer.drain()
    print(f"    Sent UUID: {fake_uuid.hex()}")

    # --- Send audio input as 320-byte PCM frames ---
    total_frames = len(pcm_input) // FRAME_SIZE
    print(f"🎙️  Sending {total_frames} audio frames ({total_frames * 20} ms)…")
    for i in range(0, len(pcm_input), FRAME_SIZE):
        chunk = pcm_input[i: i + FRAME_SIZE]
        if len(chunk) < FRAME_SIZE:
            chunk = chunk + b"\x00" * (FRAME_SIZE - len(chunk))
        writer.write(build_frame(KIND_AUDIO, chunk))
        await asyncio.sleep(0.020)  # realistic 20 ms pacing

    # Send silence to trigger flush
    silence_frame = b"\x00" * FRAME_SIZE
    print(f"    Sending 2 s of silence to trigger STT flush…")
    for _ in range(100):  # 100 × 20ms = 2 s
        writer.write(build_frame(KIND_AUDIO, silence_frame))
        await asyncio.sleep(0.020)
    await writer.drain()

    # --- Collect response audio ---
    print(f"\n⏳  Listening for response for {listen_seconds:.0f} s…")
    response_pcm = []
    deadline = asyncio.get_event_loop().time() + listen_seconds

    while asyncio.get_event_loop().time() < deadline:
        try:
            header = await asyncio.wait_for(reader.readexactly(3), timeout=0.5)
        except (asyncio.TimeoutError, asyncio.IncompleteReadError):
            continue
        except ConnectionResetError:
            print("    Bridge closed the connection")
            break

        kind = header[0]
        length = struct.unpack(">H", header[1:3])[0]
        payload = await reader.readexactly(length) if length > 0 else b""

        if kind == KIND_AUDIO:
            response_pcm.append(payload)
        elif kind == KIND_HANGUP:
            print("    Received hangup signal")
            break

    # Send hangup
    writer.write(build_frame(KIND_HANGUP, b""))
    await writer.drain()
    writer.close()

    if response_pcm:
        audio_out = b"".join(response_pcm)
        pcm_to_wav(audio_out, output_wav)
        print(f"\n✅  Response audio saved → {output_wav}")
        print(f"    Duration : {len(audio_out) / 16000:.1f} s  ({len(response_pcm)} frames)")
        print(f"    Play it  : aplay {output_wav}  OR  open it in any audio player")
    else:
        print("\n⚠️  No audio received. Is the bridge running? Is server.py running?")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="AudioSocket test client for asterisk_bridge.py")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9092)
    p.add_argument("--wav",  help="Input WAV file to send as 'caller voice'")
    p.add_argument("--text", help="Text to TTS into 'caller voice' (requires gTTS + pydub)")
    p.add_argument("--smoke", action="store_true", help="Send a synthetic beep (smoke test)")
    p.add_argument("--output", default="bridge_response.wav",
                   help="Where to save the response audio (default: bridge_response.wav)")
    p.add_argument("--listen", type=float, default=12.0,
                   help="Seconds to wait for response (default: 12)")
    args = p.parse_args()

    # Build PCM input
    if args.wav:
        print(f"📂  Loading WAV: {args.wav}")
        pcm_input = wav_to_pcm8k(args.wav)
    elif args.text:
        print(f"🗣️  Converting text to speech: \"{args.text}\"")
        pcm_input = text_to_pcm8k(args.text)
    elif args.smoke:
        print("🔔  Generating 1 s beep for smoke test")
        pcm_input = generate_beep_pcm(1000)
    else:
        print("❌  Provide --wav, --text, or --smoke")
        sys.exit(1)

    asyncio.run(run_test(args.host, args.port, pcm_input, args.output, args.listen))


if __name__ == "__main__":
    main()
