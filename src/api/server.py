import os
import json
import time
import uuid
import asyncio
import tempfile
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

from src.config.settings import HOST, PORT, LOG_LEVEL
from src.config.constants import MODEL_ID
from src.adapters.gemma import GemmaAdapter
from src.adapters.whisper import WhisperAdapter
from src.core.agent import GemmaAgent
from src.core.audio import resample_to_16k

# Initialize components
gemma_adapter = GemmaAdapter()
whisper_adapter = WhisperAdapter()
agent = GemmaAgent(gemma_adapter, whisper_adapter)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[*] Loading models...")
    gemma_adapter.load()
    whisper_adapter.load()
    print(f"[OK] Models loaded - streaming={'native' if gemma_adapter.has_async_send else 'simulated'}")
    yield
    gemma_adapter.close()

app = FastAPI(
    title="Gemma 4 Server",
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

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "engine_ready": gemma_adapter.engine is not None,
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
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
    
    # LiteRT LM requires actual Python callables for tools
    from src.tools.registry import TOOL_MAP
    resolved_tools = []
    if tools_json:
        for t in tools_json:
            name = t.get("function", {}).get("name")
            if name in TOOL_MAP:
                resolved_tools.append(TOOL_MAP[name])

    if stream:
        # For simplicity in this initial refactor, we'll implement a basic stream
        return StreamingResponse(_mock_stream(prompt), media_type="text/event-stream")
    
    agent_response = await agent.generate_text(prompt, history=history, system_msg=system_msg, tools=resolved_tools)
    
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
