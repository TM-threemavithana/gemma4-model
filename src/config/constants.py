# Audio Settings
SAMPLE_RATE_16K = 16000
SAMPLE_RATE_8K = 8000
SAMPLE_RATE_PIPER = 22050  # Default, can be updated from model config
SAMPLE_RATE_KOKORO = 24000

# Silence Detection
SILENCE_THRESHOLD = 500
SILENCE_FRAMES_NEEDED = 40  # 40 * 20ms = 0.8s
MIN_SPEECH_FRAMES = 10     # 200ms
MAX_SPEECH_FRAMES = 500    # 10s

# AudioSocket (Asterisk)
KIND_UUID = 0x00
KIND_AUDIO = 0x10
KIND_HANGUP = 0x01
KIND_ERROR = 0xFF
FRAME_SIZE = 320  # 160 samples * 2 bytes @ 8kHz (20ms)

# LLM Constants
MODEL_ID = "gemma-4-e2b-it"
DEFAULT_SYSTEM_MSG = (
    "You are Gemma, a helpful AI voice assistant. "
    "Keep your answers SHORT — 2-3 sentences max. "
    "You are speaking on a phone call, so avoid markdown or bullet points."
)

# Supported Formats
SUPPORTED_FORMATS = ["pcm", "wav"]
MAX_AUDIO_BUFFER_MS = 3000
