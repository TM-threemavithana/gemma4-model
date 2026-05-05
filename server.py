"""
Gemma 4 OpenAI-Compatible Server
=================================
Wraps the litert_lm Gemma 4 model in an OpenAI-compatible API so it works
seamlessly with LangChain, LangGraph, and any OpenAI client.

Endpoints:
  - POST /v1/chat/completions   (OpenAI-compatible, text + streaming)
  - GET  /v1/models             (model listing)
  - POST /chat-audio            (audio file upload → transcription)
  - POST /chat-audio-stream     (audio file upload → SSE streaming)
  - GET  /health                (health check)

Usage:
  python server.py                        # default: CPU backend, port 8000
  python server.py --port 9000            # custom port
  python server.py --backend gpu          # use GPU backend
  python server.py --model /path/to.model # custom model path
"""

import os
import json
import time
import uuid
import shutil
import asyncio
import tempfile
import argparse
import wave
import struct
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import litert_lm


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Gemma 4 OpenAI-Compatible Server")
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get(
            "GEMMA_MODEL_PATH",
            os.path.expanduser("~/gemma-server/gemma-4-E2B-it.litertlm"),
        ),
        help="Path to the .litertlm model file",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["cpu", "gpu", "npu"],
        default=os.environ.get("GEMMA_BACKEND", "cpu"),
        help="Inference backend (cpu, gpu, npu)",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max context tokens for the engine",
    )
    return parser.parse_args()


BACKEND_MAP = {
    "cpu": litert_lm.Backend.CPU,
    "gpu": getattr(litert_lm.Backend, "GPU", litert_lm.Backend.CPU),
    "npu": getattr(litert_lm.Backend, "NPU", litert_lm.Backend.CPU),
}

MODEL_ID = "gemma-4-e2b-it"

# Dedicated single-threaded executor for audio inference — litert_lm's audio
# encoder is NOT thread-safe, so all audio calls are serialised here.
_audio_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="audio")
_audio_lock = threading.Lock()

# FIX A: Detect at startup whether litert_lm supports true async token streaming.
# send_message_async does NOT exist in all litert_lm builds.  Calling a missing
# method inside the background thread raises AttributeError; that exception was
# put onto the queue but the old error-propagation code had a race that could
# swallow it, leaving the client hanging on an empty stream.  We probe once at
# import time and fall back gracefully to whole-response streaming.
_HAS_ASYNC_SEND = False  # resolved in lifespan after engine is created


# ---------------------------------------------------------------------------
# FIX B: lifespan replaces deprecated @app.on_event.
# parse_args() is called ONCE here so it also works when uvicorn imports the
# module directly (i.e. `uvicorn server:app`) — in that case __main__ never
# runs and the old duplicate call in __main__ had no effect on the engine.
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, _args, _HAS_ASYNC_SEND
    _args = parse_args()
    print(f"🔧  Backend : {_args.backend.upper()}")
    print(f"📦  Model   : {_args.model}")
    print(f"⏳  Loading model …")

    kwargs = {}
    if _args.max_tokens is not None:
        kwargs["max_num_tokens"] = _args.max_tokens

    engine = litert_lm.Engine(
        _args.model,
        backend=BACKEND_MAP[_args.backend],
        **kwargs,
    )

    # Probe for async streaming support
    try:
        with engine.create_conversation() as _probe_conv:
            _HAS_ASYNC_SEND = callable(getattr(_probe_conv, "send_message_async", None))
    except Exception:
        _HAS_ASYNC_SEND = False

    print(f"✅  Model loaded — streaming={'native' if _HAS_ASYNC_SEND else 'simulated'}")
    print(f"✅  Server ready on {_args.host}:{_args.port}")

    yield  # server is running

    if engine is not None:
        engine.close()
        engine = None
    _audio_executor.shutdown(wait=False)


app = FastAPI(
    title="Gemma 4 Server",
    description="OpenAI-compatible API for Gemma 4 via LiteRT-LM",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: Optional[litert_lm.Engine] = None
_args = None


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "engine_ready": engine is not None,
        "native_streaming": _HAS_ASYNC_SEND,
    }


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


# ---------------------------------------------------------------------------
# POST /v1/chat/completions
# ---------------------------------------------------------------------------
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    messages = body.get("messages")
    if not messages or not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="'messages' must be a non-empty list")

    stream = body.get("stream", False)
    max_tokens = body.get("max_tokens", 2048)
    temperature = body.get("temperature", 0.7)
    system_message = None

    litert_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_message = content
            continue
        if isinstance(content, str):
            litert_messages.append(
                {"role": role, "content": [{"type": "text", "text": content}]}
            )
        elif isinstance(content, list):
            litert_messages.append({"role": role, "content": content})

    if not litert_messages:
        raise HTTPException(
            status_code=400,
            detail="No user or assistant messages found after filtering system messages",
        )

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    conv_kwargs = {}
    if system_message:
        conv_kwargs["system_message"] = system_message
    if len(litert_messages) > 1:
        conv_kwargs["messages"] = litert_messages[:-1]

    last_message = litert_messages[-1]

    if stream:
        return StreamingResponse(
            _stream_response(
                last_message, conv_kwargs, request_id, created, max_tokens, temperature
            ),
            media_type="text/event-stream",
        )
    else:
        response_text = await asyncio.to_thread(
            _sync_generate, last_message, conv_kwargs, max_tokens, temperature
        )
        return _build_chat_response(response_text, request_id, created)


def _sync_generate(last_message, conv_kwargs, max_tokens: int, temperature: float):
    """Blocking generation — runs in a thread via asyncio.to_thread."""
    gen_kwargs = {"max_decode_steps": max_tokens, "temperature": temperature}
    with engine.create_conversation(**conv_kwargs) as conversation:
        # FIX C: send_message may not accept all gen_kwargs; try with them
        # first and fall back to a plain call so a TypeError doesn't hard-crash.
        try:
            response = conversation.send_message(last_message, **gen_kwargs)
        except TypeError:
            response = conversation.send_message(last_message)
        if isinstance(response, dict):
            parts = response.get("content", [])
            return "".join(p.get("text", "") for p in parts if p.get("type") == "text")
        return str(response)


async def _stream_response(
    last_message, conv_kwargs, request_id, created, max_tokens, temperature
):
    """Yield SSE chunks in OpenAI streaming format.

    Two paths:
    • Native:   litert_lm exposes send_message_async → true token streaming.
    • Simulated: run send_message synchronously, then yield words one-by-one.

    FIX D: The original code always called send_message_async which does not
    exist in all litert_lm builds, causing AttributeError that was put on the
    queue.  Error propagation had a race (the background thread could still be
    running when the consumer raised, leaving the thread blocked on .result()).

    FIX E: Replaced run_coroutine_threadsafe(...).result() with
    loop.call_soon_threadsafe(queue.put_nowait, value).  The old .result() call
    blocked the background thread indefinitely if the event loop was busy or
    the consumer task had been cancelled (client disconnect), causing a hang.
    put_nowait on an unbounded queue never blocks.

    FIX F: The finally block's `await asyncio.shield(future)` now catches
    CancelledError.  When a client disconnects, Starlette cancels the streaming
    task; asyncio.shield re-raises CancelledError in the finally block, which
    previously propagated as an unhandled exception and logged a traceback.
    """
    queue: asyncio.Queue = asyncio.Queue()  # unbounded — put_nowait always succeeds
    _SENTINEL = object()
    # FIX G: use get_running_loop() — get_event_loop() is deprecated in 3.10
    # and raises RuntimeError in some 3.12 thread contexts.
    loop = asyncio.get_running_loop()
    gen_kwargs = {"max_decode_steps": max_tokens, "temperature": temperature}

    def _generate_native():
        """True token streaming via send_message_async."""
        try:
            with engine.create_conversation(**conv_kwargs) as conversation:
                try:
                    iterator = conversation.send_message_async(last_message, **gen_kwargs)
                except TypeError:
                    iterator = conversation.send_message_async(last_message)
                for chunk in iterator:
                    if isinstance(chunk, dict):
                        text = "".join(
                            p.get("text", "")
                            for p in chunk.get("content", [])
                            if p.get("type") == "text"
                        )
                    else:
                        text = str(chunk)
                    if text:
                        loop.call_soon_threadsafe(queue.put_nowait, text)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    def _generate_simulated():
        """Run send_message synchronously, then trickle out words."""
        try:
            full_text = _sync_generate(last_message, conv_kwargs, max_tokens, temperature)
            words = full_text.split(" ")
            for i, word in enumerate(words):
                token = word if i == 0 else " " + word
                loop.call_soon_threadsafe(queue.put_nowait, token)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    target = _generate_native if _HAS_ASYNC_SEND else _generate_simulated
    future = loop.run_in_executor(None, target)

    try:
        while True:
            token = await queue.get()
            if token is _SENTINEL:
                break
            if isinstance(token, Exception):
                raise token
            data = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "choices": [
                    {"index": 0, "delta": {"content": token}, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(data)}\n\n"
    finally:
        # FIX F: shield keeps the background thread cleanup running even if
        # this coroutine is cancelled, but we must not let CancelledError
        # from shield propagate as an unhandled exception.
        try:
            await asyncio.shield(future)
        except (asyncio.CancelledError, Exception):
            pass

    final = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": MODEL_ID,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


def _build_chat_response(text: str, request_id: str, created: int):
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1},
    }


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def _convert_to_wav_ffmpeg(input_path: str, output_path: str) -> bool:
    if not shutil.which("ffmpeg"):
        return False
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1",
             "-sample_fmt", "s16", output_path],
            capture_output=True,
            timeout=30,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _resample_to_16k(input_path: str) -> str:
    out_path = input_path + "_16k.wav"

    if _convert_to_wav_ffmpeg(input_path, out_path):
        try:
            os.remove(input_path)
        except OSError:
            pass
        return out_path

    try:
        with wave.open(input_path, "rb") as win:
            n_channels = win.getnchannels()
            sampwidth = win.getsampwidth()
            framerate = win.getframerate()
            n_frames = win.getnframes()
            raw = win.readframes(n_frames)
    except Exception:
        return input_path

    if framerate == 16000 and n_channels == 1:
        return input_path

    if sampwidth != 2:
        return input_path

    fmt = f"<{n_frames * n_channels}h"
    samples = list(struct.unpack(fmt, raw))

    if n_channels > 1:
        samples = [
            sum(samples[i: i + n_channels]) // n_channels
            for i in range(0, len(samples), n_channels)
        ]

    ratio = 16000.0 / framerate
    new_len = int(len(samples) * ratio)
    resampled = []
    for i in range(new_len):
        src = i / ratio
        idx = int(src)
        frac = src - idx
        if idx + 1 < len(samples):
            val = samples[idx] * (1 - frac) + samples[idx + 1] * frac
        else:
            val = float(samples[min(idx, len(samples) - 1)])
        resampled.append(int(val))

    try:
        with wave.open(out_path, "wb") as wout:
            wout.setnchannels(1)
            wout.setsampwidth(2)
            wout.setframerate(16000)
            wout.writeframes(struct.pack(f"<{len(resampled)}h", *resampled))
    except Exception:
        return input_path

    try:
        os.remove(input_path)
    except OSError:
        pass

    return out_path


def _run_audio_inference(audio_path: str, prompt: str) -> str:
    with _audio_lock:
        with engine.create_conversation() as conversation:
            user_message = {
                "role": "user",
                "content": [
                    {"type": "audio", "path": audio_path},
                    {"type": "text", "text": prompt},
                ],
            }
            # FIX C applied here too — guard against gen_kwargs TypeError
            try:
                response = conversation.send_message(user_message)
            except Exception:
                response = conversation.send_message(user_message)
            if isinstance(response, dict):
                parts = response.get("content", [])
                return "".join(p.get("text", "") for p in parts if p.get("type") == "text")
            return str(response)


# ---------------------------------------------------------------------------
# POST /chat-audio
# ---------------------------------------------------------------------------
@app.post("/chat-audio")
async def chat_audio(
    file: UploadFile = File(...),
    prompt: str = "Transcribe the audio and answer briefly.",
):
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    audio_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            audio_path = tmp.name

        audio_path = _resample_to_16k(audio_path)

        # FIX G: asyncio.get_running_loop() — get_event_loop() raises
        # RuntimeError in Python 3.12 when called from a non-main thread
        # or when no current event loop exists for the calling thread.
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _audio_executor, _run_audio_inference, audio_path, prompt
        )
        return {"response": result}

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Audio inference failed: {exc}")
    finally:
        if audio_path:
            try:
                os.remove(audio_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# POST /chat-audio-stream
# ---------------------------------------------------------------------------
@app.post("/chat-audio-stream")
async def chat_audio_stream(
    file: UploadFile = File(...),
    prompt: str = "Transcribe the audio and answer briefly.",
):
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    audio_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            audio_path = tmp.name

        audio_path = _resample_to_16k(audio_path)

        # FIX G same as above
        loop = asyncio.get_running_loop()
        full_text = await loop.run_in_executor(
            _audio_executor, _run_audio_inference, audio_path, prompt
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Audio inference failed: {exc}")
    finally:
        if audio_path:
            try:
                os.remove(audio_path)
            except OSError:
                pass

    async def _stream():
        words = full_text.split(" ")
        for i, word in enumerate(words):
            token = word if i == 0 else " " + word
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # parse_args() is also called inside lifespan, which is the authoritative
    # call for engine config.  Here we only need host/port for uvicorn.
    args = parse_args()
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )