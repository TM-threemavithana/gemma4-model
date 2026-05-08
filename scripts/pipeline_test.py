# %% [markdown]
# # Gemma 4 Voice Calling Agent — Pipeline Test Notebook
# 
# Test each step of the pipeline in isolation.
# 
# ### 📦 Prerequisites
# If you are running this for the first time, run the cell below to ensure all dependencies are installed.

# %%
import sys
try:
    import piper
    import faster_whisper
    import httpx
    print("✅ All dependencies found.")
except ImportError:
    print("📦 Installing missing dependencies...")
    # Use %pip if in Jupyter, otherwise use subprocess
    try:
        get_ipython()
        %pip install piper-tts faster-whisper httpx pydub gtts --quiet
    except NameError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "piper-tts", "faster-whisper", "httpx", "pydub", "gtts"])
    print("✅ Done! Please RESTART YOUR KERNEL/SCRIPT now.")

# %% [markdown]
# ## Configuration

# %%
import os, sys, time, json, io, wave, struct, tempfile, subprocess
import httpx
import IPython.display as ipd

GEMMA_URL = "http://localhost:8000"
BRIDGE_HOST = "127.0.0.1"
BRIDGE_PORT = 9092
SYSTEM_MSG = (
    "You are Gemma, a helpful AI voice assistant. "
    "Keep your answers SHORT — 2-3 sentences max."
)
print("✅ Configuration loaded")

# %% [markdown]
# ---
# ## Step 1: Server Health Check
# Verify the Gemma 4 `server.py` is running and responsive.

# %%
def check_server_health(url=GEMMA_URL):
    """Check if the Gemma 4 server is running."""
    try:
        r = httpx.get(f"{url}/health", timeout=5.0)
        r.raise_for_status()
        data = r.json()
        print(f"✅ Server is HEALTHY")
        for k, v in data.items():
            print(f"   {k}: {v}")
        return data
    except Exception as e:
        print(f"❌ Server unreachable: {e}")
        print(f"   Start it with: python server.py")
        return None

health = check_server_health()

# %% [markdown]
# ## Step 2: List Models
# Query the OpenAI-compatible `/v1/models` endpoint.

# %%
def list_models(url=GEMMA_URL):
    """List available models from the server."""
    try:
        r = httpx.get(f"{url}/v1/models", timeout=5.0)
        r.raise_for_status()
        data = r.json()
        print("✅ Available models:")
        for m in data.get("data", []):
            print(f"   - {m['id']} (owned_by: {m['owned_by']})")
        return data
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None

models = list_models()

# %% [markdown]
# ---
# ## Step 3: Text Chat Completion (Non-Streaming)
# Send a text prompt to the LLM and get a complete response.

# %%
def chat_text(prompt, system_msg=SYSTEM_MSG, url=GEMMA_URL, max_tokens=200):
    """Send a text chat completion request (non-streaming)."""
    payload = {
        "model": "gemma-4-e2b-it",
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
    }
    t0 = time.time()
    r = httpx.post(f"{url}/v1/chat/completions", json=payload, timeout=60.0)
    r.raise_for_status()
    elapsed = time.time() - t0
    data = r.json()
    reply = data["choices"][0]["message"]["content"]
    print(f"✅ Response ({elapsed:.2f}s):")
    print(f"   {reply}")
    return reply

reply = chat_text("Hello! What can you do?")

# %% [markdown]
# ---
# ## Step 4: Streaming Chat Completion (SSE)
# Test real-time token-by-token streaming from the server.

# %%
def chat_stream(prompt, system_msg=SYSTEM_MSG, url=GEMMA_URL, max_tokens=200):
    """Send a streaming chat completion request and print tokens live."""
    payload = {
        "model": "gemma-4-e2b-it",
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }
    t0 = time.time()
    full = []
    with httpx.stream("POST", f"{url}/v1/chat/completions", json=payload, timeout=60.0) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            chunk = json.loads(data_str)
            token = chunk["choices"][0].get("delta", {}).get("content", "")
            if token:
                full.append(token)
                print(token, end="", flush=True)
    elapsed = time.time() - t0
    print(f"\n\n✅ Streaming complete ({elapsed:.2f}s, {len(full)} tokens)")
    return "".join(full)

streamed = chat_stream("Tell me a fun fact about space.")

# %% [markdown]
# ---
# ## Step 5: Multi-Turn Conversation
# Test conversation history tracking.

# %%
def multi_turn_chat(turns, system_msg=SYSTEM_MSG, url=GEMMA_URL):
    """Run a multi-turn conversation and print each exchange."""
    history = [{"role": "system", "content": system_msg}]
    for i, user_msg in enumerate(turns, 1):
        history.append({"role": "user", "content": user_msg})
        payload = {
            "model": "gemma-4-e2b-it",
            "messages": history,
            "max_tokens": 200,
            "temperature": 0.7,
            "stream": False,
        }
        t0 = time.time()
        r = httpx.post(f"{url}/v1/chat/completions", json=payload, timeout=60.0)
        r.raise_for_status()
        reply = r.json()["choices"][0]["message"]["content"]
        elapsed = time.time() - t0
        print(f"--- Turn {i} ({elapsed:.2f}s) ---")
        print(f"👤 User: {user_msg}")
        print(f"🤖 Gemma: {reply}\n")
        history.append({"role": "assistant", "content": reply})
    return history

history = multi_turn_chat([
    "My name is Alex.",
    "What is 15 * 23?",
    "What's my name?",
])

# %% [markdown]
# ---
# ## Step 6: Whisper STT (Speech-to-Text)
# Test the `faster-whisper` transcription locally. Run this in WSL2 if that's where whisper is installed.

# %%
def test_whisper_stt(wav_path=None):
    """Test Whisper STT. If no wav_path, generate a test tone."""
    if wav_path is None:
        # Generate a short WAV with silence (just to test loading)
        import math
        wav_path = os.path.join(tempfile.gettempdir(), "test_whisper.wav")
        sr = 16000
        duration = 2.0
        samples = [int(8000 * math.sin(2 * math.pi * 440 * i / sr)) for i in range(int(sr * duration))]
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))
        print(f"   Generated test WAV: {wav_path}")

    try:
        from faster_whisper import WhisperModel
        print("⏳ Loading Whisper model (base, CPU)...")
        t0 = time.time()
        model = WhisperModel("base", device="cpu", compute_type="int8")
        print(f"✅ Whisper loaded in {time.time()-t0:.1f}s")

        t0 = time.time()
        segments, info = model.transcribe(wav_path, beam_size=5)
        text = " ".join(s.text for s in segments).strip()
        elapsed = time.time() - t0
        print(f"✅ Transcription ({elapsed:.2f}s): '{text}'")
        print(f"   Language: {info.language} (prob: {info.language_probability:.2f})")
        return text
    except ImportError:
        print("❌ faster-whisper not installed.")
        print("   Install: pip install faster-whisper")
        return None

transcript = test_whisper_stt()

# %% [markdown]
# ---
# ## Step 7: Piper TTS (Text-to-Speech)
# Test local Piper TTS synthesis.

# %%
def test_piper_tts(text="Hello, this is a test of the Piper text to speech engine.", model_path=None):
    """Test Piper TTS synthesis."""
    if model_path is None:
        model_path = os.path.expanduser("~/piper-models/en_US-amy-medium.onnx")

    try:
        from piper.voice import PiperVoice
    except ImportError:
        print("❌ piper-tts not installed. Run: pip install piper-tts")
        return None

    config_path = model_path + ".json"
    if not os.path.exists(model_path):
        print(f"❌ Piper model not found: {model_path}")
        return None

    print(f"⏳ Loading Piper model: {os.path.basename(model_path)}")
    t0 = time.time()
    voice = PiperVoice.load(model_path, config_path=config_path)
    print(f"✅ Piper loaded in {time.time()-t0:.1f}s (sample rate: {voice.config.sample_rate} Hz)")

    print(f"⏳ Synthesizing: '{text}'")
    t0 = time.time()
    raw_audio = b"".join(chunk.audio_int16_bytes for chunk in voice.synthesize(text))
    elapsed = time.time() - t0
    duration = len(raw_audio) / (voice.config.sample_rate * 2)
    print(f"✅ TTS done in {elapsed*1000:.0f}ms — audio duration: {duration:.1f}s")

    # Save as WAV for playback
    out_path = os.path.join(tempfile.gettempdir(), "piper_test.wav")
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(voice.config.sample_rate)
        wf.writeframes(raw_audio)
    print(f"   Saved: {out_path}")

    try:
        return ipd.Audio(out_path)
    except Exception:
        return out_path

audio_widget = test_piper_tts()
audio_widget

# %% [markdown]
# ---
# ## Step 8: Audio Upload to Server (`/chat-audio`)
# Upload a WAV file to the Gemma 4 server's audio endpoint.

# %%
def test_audio_upload(wav_path=None, prompt="Transcribe this audio.", url=GEMMA_URL):
    """Upload a WAV to /chat-audio and get the response."""
    if wav_path is None:
        wav_path = os.path.join(tempfile.gettempdir(), "piper_test.wav")
        if not os.path.exists(wav_path):
            print("❌ No WAV file. Run Step 7 (Piper TTS) first to generate one.")
            return None

    print(f"⏳ Uploading {os.path.basename(wav_path)} to {url}/chat-audio ...")
    t0 = time.time()
    with open(wav_path, "rb") as f:
        r = httpx.post(
            f"{url}/chat-audio",
            files={"file": (os.path.basename(wav_path), f, "audio/wav")},
            data={"prompt": prompt},
            timeout=60.0,
        )
    r.raise_for_status()
    elapsed = time.time() - t0
    data = r.json()
    print(f"✅ Response ({elapsed:.2f}s):")
    print(f"   {data.get('response', data)}")
    return data

audio_response = test_audio_upload()

# %% [markdown]
# ---
# ## Step 9: Full Local Pipeline (STT → LLM → TTS)
# Run the complete pipeline using local components — no Asterisk needed.

# %%
def full_pipeline(wav_input=None, text_input=None, url=GEMMA_URL):
    """Run STT → LLM → TTS end-to-end."""
    timings = {}

    # --- Step A: Get input text ---
    if text_input:
        user_text = text_input
        print(f"📝 Input text: '{user_text}'")
    elif wav_input:
        print("🎙️ Step A: Whisper STT...")
        t0 = time.time()
        from faster_whisper import WhisperModel
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(wav_input, beam_size=5)
        user_text = " ".join(s.text for s in segments).strip()
        timings["stt"] = time.time() - t0
        print(f"   Transcribed ({timings['stt']:.2f}s): '{user_text}'")
    else:
        user_text = "What is the capital of France?"
        print(f"📝 Using default: '{user_text}'")

    # --- Step B: Gemma LLM ---
    print("🤖 Step B: Gemma 4 LLM...")
    t0 = time.time()
    payload = {
        "model": "gemma-4-e2b-it",
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_text},
        ],
        "max_tokens": 200, "temperature": 0.7, "stream": False,
    }
    r = httpx.post(f"{url}/v1/chat/completions", json=payload, timeout=60.0)
    r.raise_for_status()
    reply = r.json()["choices"][0]["message"]["content"]
    timings["llm"] = time.time() - t0
    print(f"   Reply ({timings['llm']:.2f}s): '{reply}'")

    # --- Step C: Piper TTS ---
    print("🔊 Step C: Piper TTS...")
    try:
        from piper.voice import PiperVoice
        model_path = os.path.expanduser("~/piper-models/en_US-amy-medium.onnx")
        voice = PiperVoice.load(model_path, config_path=model_path + ".json")
        t0 = time.time()
        raw = b"".join(c.audio_int16_bytes for c in voice.synthesize(reply))
        timings["tts"] = time.time() - t0
        out = os.path.join(tempfile.gettempdir(), "pipeline_output.wav")
        with wave.open(out, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(voice.config.sample_rate)
            wf.writeframes(raw)
        print(f"   Audio ({timings['tts']*1000:.0f}ms): {out}")
    except Exception as e:
        print(f"   ⚠️ TTS skipped: {e}")
        out = None

    # --- Summary ---
    total = sum(timings.values())
    print(f"\n{'='*50}")
    print(f"📊 Pipeline Timing Summary:")
    for step, t in timings.items():
        print(f"   {step.upper():>4}: {t*1000:>6.0f} ms")
    print(f"   {'TOTAL':>4}: {total*1000:>6.0f} ms")
    print(f"{'='*50}")

    if out:
        try:
            return ipd.Audio(out)
        except Exception:
            return out

result = full_pipeline(text_input="Tell me a joke.")

# %% [markdown]
# ---
# ## Step 10: Asterisk Service Status
# Check if Asterisk is running in WSL2.

# %%
def check_asterisk():
    """Check Asterisk service status in WSL2."""
    try:
        r = subprocess.run(
            ["wsl", "-u", "root", "service", "asterisk", "status"],
            capture_output=True, text=True, timeout=10
        )
        output = r.stdout + r.stderr
        running = "running" in output.lower() or r.returncode == 0
        print(f"{'✅' if running else '❌'} Asterisk: {'RUNNING' if running else 'STOPPED'}")
        print(f"   {output.strip()[:200]}")
        return running
    except Exception as e:
        print(f"❌ Could not check Asterisk: {e}")
        return False

asterisk_ok = check_asterisk()

# %% [markdown]
# ---
# ## Step 11: AudioSocket Simulation
# Simulate a phone call by connecting directly to the bridge (no phone needed).
# **Requires**: `asterisk_bridge.py` running + `server.py` running.

# %%
import asyncio

async def simulate_call(text="Hello Gemma, what is two plus two?", host=BRIDGE_HOST, port=BRIDGE_PORT, listen_s=12.0):
    """Simulate an AudioSocket call to the bridge."""
    import math, uuid

    KIND_UUID, KIND_AUDIO, KIND_HANGUP = 0x00, 0x10, 0x01
    FRAME_SIZE = 320

    # Generate speech PCM at 8kHz
    print(f"🗣️ Generating speech for: '{text}'")
    try:
        from piper.voice import PiperVoice
        import audioop
        mp = os.path.expanduser("~/piper-models/en_US-amy-medium.onnx")
        v = PiperVoice.load(mp, config_path=mp+".json")
        raw = b"".join(c.audio_int16_bytes for c in v.synthesize(text))
        pcm, _ = audioop.ratecv(raw, 2, 1, v.config.sample_rate, 8000, None)
    except Exception:
        # Fallback: sine wave
        sr = 8000
        samples = [int(8000*math.sin(2*math.pi*440*i/sr)) for i in range(sr*2)]
        pcm = struct.pack(f"<{len(samples)}h", *samples)

    def build_frame(kind, payload):
        return bytes([kind]) + struct.pack(">H", len(payload)) + payload

    print(f"🔌 Connecting to bridge at {host}:{port}...")
    reader, writer = await asyncio.open_connection(host, port)
    print("✅ Connected")

    # Send UUID
    writer.write(build_frame(KIND_UUID, uuid.uuid4().bytes))
    await writer.drain()

    # Wait for greeting (collect for 5s)
    print("⏳ Receiving greeting...")
    greeting_pcm = []
    deadline = asyncio.get_event_loop().time() + 5.0
    while asyncio.get_event_loop().time() < deadline:
        try:
            hdr = await asyncio.wait_for(reader.readexactly(3), timeout=0.3)
        except (asyncio.TimeoutError, asyncio.IncompleteReadError):
            continue
        kind = hdr[0]
        length = struct.unpack(">H", hdr[1:3])[0]
        payload = await reader.readexactly(length) if length > 0 else b""
        if kind == KIND_AUDIO:
            greeting_pcm.append(payload)

    if greeting_pcm:
        g_raw = b"".join(greeting_pcm)
        g_path = "nb_greeting.wav"
        with wave.open(g_path, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(8000)
            wf.writeframes(g_raw)
        print(f"🔊 Greeting received ({len(g_raw)/16000:.1f}s) → {g_path}")

    # Send audio frames
    n_frames = len(pcm) // FRAME_SIZE
    print(f"🎙️ Sending {n_frames} frames ({n_frames*20}ms)...")
    for i in range(0, len(pcm), FRAME_SIZE):
        chunk = pcm[i:i+FRAME_SIZE]
        if len(chunk) < FRAME_SIZE:
            chunk += b"\x00" * (FRAME_SIZE - len(chunk))
        writer.write(build_frame(KIND_AUDIO, chunk))
        await asyncio.sleep(0.020)

    # Silence to trigger flush
    silence = b"\x00" * FRAME_SIZE
    for _ in range(100):
        writer.write(build_frame(KIND_AUDIO, silence))
        await asyncio.sleep(0.020)
    await writer.drain()
    print("   Sent 2s silence to trigger STT flush")

    # Collect response
    print(f"⏳ Listening for response ({listen_s}s)...")
    resp_pcm = []
    deadline = asyncio.get_event_loop().time() + listen_s
    while asyncio.get_event_loop().time() < deadline:
        try:
            hdr = await asyncio.wait_for(reader.readexactly(3), timeout=0.5)
        except (asyncio.TimeoutError, asyncio.IncompleteReadError):
            continue
        except ConnectionResetError:
            break
        kind = hdr[0]
        length = struct.unpack(">H", hdr[1:3])[0]
        payload = await reader.readexactly(length) if length > 0 else b""
        if kind == KIND_AUDIO:
            resp_pcm.append(payload)
        elif kind == KIND_HANGUP:
            break

    # Hangup
    writer.write(build_frame(KIND_HANGUP, b""))
    await writer.drain()
    writer.close()

    if resp_pcm:
        r_raw = b"".join(resp_pcm)
        r_path = "nb_response.wav"
        with wave.open(r_path, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(8000)
            wf.writeframes(r_raw)
        print(f"✅ Response audio ({len(r_raw)/16000:.1f}s) → {r_path}")
        try:
            return ipd.Audio(r_path)
        except Exception:
            return r_path
    else:
        print("⚠️ No response audio received.")
        return None

# Run the simulation (use await in Jupyter, asyncio.run() in script)
try:
    get_ipython()
    # We're in Jupyter
    result = await simulate_call("Hello Gemma, what is two plus two?")
    result
except NameError:
    result = asyncio.run(simulate_call("Hello Gemma, what is two plus two?"))
