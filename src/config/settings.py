import os
import sys
from pathlib import Path

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Model Paths
DEFAULT_MODEL = os.environ.get("GEMMA_MODEL_PATH")
if not DEFAULT_MODEL:
    candidate = os.path.expanduser("~/gemma-server/gemma-4-E2B-it.litertlm")
    if os.path.exists(candidate):
        DEFAULT_MODEL = candidate
    elif sys.platform == "win32":
        wsl_fallback = r"\\wsl.localhost\Ubuntu\home\tharuka\gemma-server\gemma-4-E2B-it.litertlm"
        if os.path.exists(wsl_fallback):
            DEFAULT_MODEL = wsl_fallback

MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL or "assets/models/gemma-4-E2B-it.litertlm")
PIPER_MODEL_PATH = os.getenv("PIPER_MODEL_PATH", os.path.expanduser("~/piper-models/en_US-amy-medium.onnx"))

# Kokoro TTS Settings
KOKORO_MODEL_DIR = os.getenv("KOKORO_MODEL_DIR", os.path.expanduser("~/kokoro-models"))
KOKORO_MODEL_FILE = os.path.join(KOKORO_MODEL_DIR, "kokoro-v1.0.onnx")
KOKORO_VOICES_FILE = os.path.join(KOKORO_MODEL_DIR, "voices-v1.0.bin")
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_sarah")
KOKORO_SPEED = float(os.getenv("KOKORO_SPEED", "1.0"))

# TTS Engine selection: "kokoro" or "piper"
TTS_ENGINE = os.getenv("TTS_ENGINE", "kokoro")

# Network Settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
ASTERISK_PORT = int(os.getenv("ASTERISK_PORT", 9092))
GEMMA_URL = os.getenv("GEMMA_URL", f"http://localhost:{PORT}")

# Inference Settings
BACKEND = os.getenv("GEMMA_BACKEND", "cpu")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 2048))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Data Persistence
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = str(DATA_DIR / "calls.db")
TEMP_AUDIO_DIR = str(DATA_DIR / "temp_audio")
