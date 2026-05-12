import argparse
import asyncio
import logging
import threading
import httpx
from src.bridge.socket import AudioSocketBridge
from src.config.settings import HOST, PORT, ASTERISK_PORT, LOG_LEVEL, GEMMA_URL
from src.storage.database import init_db
from src.core import post_processing
from shared.db import startup as db_startup, shutdown as db_shutdown

def setup_logging():
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

async def my_gemma_summarise(text):
    """Enrichment: Summarize conversation."""
    try:
        async with httpx.AsyncClient() as client:
            payload = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant. Summarize the following conversation transcript into a single, concise sentence focusing on the user's intent."},
                    {"role": "user", "content": text}
                ]
            }
            resp = await client.post(f"{GEMMA_URL}/v1/chat/completions", json=payload, timeout=30)
            return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Summary failed: {e}"

async def my_gemma_sentiment(text):
    """Enrichment: Analyze sentiment."""
    try:
        async with httpx.AsyncClient() as client:
            payload = {
                "messages": [
                    {"role": "system", "content": "Analyze the sentiment of the following conversation transcript. Respond with ONLY one word: Positive, Neutral, or Negative."},
                    {"role": "user", "content": text}
                ]
            }
            resp = await client.post(f"{GEMMA_URL}/v1/chat/completions", json=payload, timeout=15)
            return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Sentiment failed: {e}"

async def run_bridge():
    # Configure post-processing
    post_processing.summarise_fn = my_gemma_summarise
    post_processing.sentiment_fn = my_gemma_sentiment
    
    bridge = AudioSocketBridge()
    await bridge.start(ASTERISK_PORT)

def run_server():
    import uvicorn
    from src.api.server import app
    uvicorn.run(app, host=HOST, port=PORT, log_level=LOG_LEVEL.lower())

def main():
    setup_logging()
    init_db()
    parser = argparse.ArgumentParser(description="Gemma 4 Multi-Interface AI")
    parser.add_argument("--mode", choices=["all", "api", "bridge"], default="all")
    args = parser.parse_args()

    if args.mode == "api":
        run_server()
    elif args.mode == "bridge":
        async def run_bridge_with_db():
            await db_startup()
            try:
                await run_bridge()
            finally:
                await db_shutdown()
        asyncio.run(run_bridge_with_db())
    else:
        # Run both in separate threads/loops
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        async def run_all_with_db():
            await db_startup()
            try:
                await run_bridge()
            finally:
                await db_shutdown()

        try:
            asyncio.run(run_all_with_db())
        except KeyboardInterrupt:
            print("Shutting down...")

if __name__ == "__main__":
    main()
