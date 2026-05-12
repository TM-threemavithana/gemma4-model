import os
import json
import time
import uuid
import asyncio
import tempfile
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Request, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

from src.config.settings import HOST, PORT, LOG_LEVEL
from src.config.constants import MODEL_ID
from src.core.instances import agent, gemma_adapter, whisper_adapter

from src.bridge.twilio import TwilioStreamBridge

twilio_bridge = TwilioStreamBridge()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[*] Loading models...")
    gemma_adapter.load()
    whisper_adapter.load()
    twilio_bridge.load()
    print(f"[OK] Models loaded - streaming={'native' if gemma_adapter.has_async_send else 'simulated'}")
    yield
    gemma_adapter.close()

app = FastAPI(
    title="Gemma 4 Server",
    version="1.0.0",
    lifespan=lifespan,
)

@app.websocket("/twiliostream")
async def twilio_stream_endpoint(websocket: WebSocket):
    await twilio_bridge.handle_websocket(websocket)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "engine_ready": gemma_adapter.engine is not None,
    }

agent_lock = asyncio.Lock()

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    async with agent_lock:
        body = await request.json()
        messages = body.get("messages", [])
        if not messages:
            raise HTTPException(status_code=400, detail="'messages' must be a non-empty list")
    
        stream = body.get("stream", False)
        # Simplified logic for now, using agent.generate_text
        prompt = messages[-1].get("content", "")
        history = messages[:-1]
        
        # Handle system message separately if present
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), None)
        history = [m for m in history if m["role"] != "system"]
    
        tools_json = body.get("tools")

        # LiteRT requires actual Python callables — wrap async tools in sync thread-safe runners
        import threading
        from src.tools.registry import TOOL_MAP

        def make_sync_wrapper(async_func):
            """Runs an async tool in a dedicated thread+event loop so LiteRT can call it synchronously.
            functools.wraps copies the original signature so inspect.signature() sees the real params."""
            import functools
            @functools.wraps(async_func)
            def sync_wrapper(*args, **kwargs):
                result_holder = [None]
                exc_holder = [None]
                def run():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result_holder[0] = loop.run_until_complete(async_func(*args, **kwargs))
                    except Exception as e:
                        exc_holder[0] = e
                    finally:
                        loop.close()
                t = threading.Thread(target=run)
                t.start()
                t.join()
                if exc_holder[0]:
                    raise exc_holder[0]
                return result_holder[0]
            return sync_wrapper

        resolved_tools = []
        if tools_json:
            for t in tools_json:
                name = t.get("function", {}).get("name")
                if name in TOOL_MAP:
                    func = TOOL_MAP[name]
                    if asyncio.iscoroutinefunction(func):
                        resolved_tools.append(make_sync_wrapper(func))
                    else:
                        resolved_tools.append(func)

        agent_response = await agent.generate_text(
            prompt, history=history, system_msg=system_msg,
            tools=resolved_tools if resolved_tools else None
        )
        
        # Handle the response structure
        message_content = ""
        tool_calls = None
        
        if isinstance(agent_response, dict):
            parts = agent_response.get("content", [])
            message_content = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
            tool_calls = agent_response.get("tool_calls")
        else:
            message_content = str(agent_response)
    
        message = {"role": "assistant", "content": message_content}
        if tool_calls:
            message["tool_calls"] = tool_calls
    
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [{"message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
        }

async def _mock_stream(prompt):
    # Placeholder for actual streaming logic
    yield f"data: {json.dumps({'choices': [{'delta': {'content': 'Streaming not yet fully refactored.'}}]})}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/chat-audio")
async def chat_audio(
    file: UploadFile = File(...),
    prompt: str = "Transcribe the audio and answer briefly.",
):
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    audio_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            audio_path = tmp.name

        audio_path = resample_to_16k(audio_path)
        result = await agent.generate_from_audio(audio_path, prompt)
        return {"response": result}
    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level=LOG_LEVEL.lower())
