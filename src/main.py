import argparse
import asyncio
import logging
import threading
import uvicorn
from src.api.server import app
from src.bridge.socket import AudioSocketBridge
from src.config.settings import HOST, PORT, ASTERISK_PORT, LOG_LEVEL

def setup_logging():
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

async def run_bridge():
    bridge = AudioSocketBridge()
    await bridge.start(ASTERISK_PORT)

def run_server():
    uvicorn.run(app, host=HOST, port=PORT, log_level=LOG_LEVEL.lower())

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Gemma 4 Multi-Interface AI")
    parser.add_argument("--mode", choices=["all", "api", "bridge"], default="all")
    args = parser.parse_args()

    if args.mode == "api":
        run_server()
    elif args.mode == "bridge":
        asyncio.run(run_bridge())
    else:
        # Run both in separate threads/loops
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        try:
            asyncio.run(run_bridge())
        except KeyboardInterrupt:
            print("Shutting down...")

if __name__ == "__main__":
    main()
