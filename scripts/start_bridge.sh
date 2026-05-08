#!/usr/bin/env bash
# =============================================================================
# start_bridge.sh — Start the Asterisk AudioSocket Bridge in WSL2
# =============================================================================
# Run this script INSIDE WSL2 (not Windows PowerShell).
# Make executable: chmod +x start_bridge.sh
# Run: ./start_bridge.sh
#
# Prerequisites (see SETUP.md for full guide):
#   1. python3, pip, faster-whisper, piper-tts, httpx installed
#   2. Piper voice model downloaded to ~/piper-models/
#   3. server.py (Gemma 4) must already be running on Windows :8000
#   4. Asterisk must be running: sudo systemctl start asterisk
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE_PY="$SCRIPT_DIR/asterisk_bridge.py"

# --- Piper TTS model path ---
PIPER_MODEL="${PIPER_MODEL:-$HOME/piper-models/en_US-amy-medium.onnx}"

# --- Check Piper model exists ---
if [ ! -f "$PIPER_MODEL" ]; then
    echo "❌  Piper TTS model not found: $PIPER_MODEL"
    echo ""
    echo "   Download it with:"
    echo "     mkdir -p ~/piper-models"
    echo "     wget -O ~/piper-models/en_US-amy-medium.onnx \\"
    echo "       https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx"
    echo "     wget -O ~/piper-models/en_US-amy-medium.onnx.json \\"
    echo "       https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx.json"
    echo ""
    exit 1
fi
echo "✅  Piper model found: $(basename $PIPER_MODEL)"

# --- Detect WSL2 Windows host IP (to reach server.py running on Windows) ---
# When running bridge inside WSL2, Windows is at the WSL2 gateway IP.
# The Gemma 4 server.py runs on Windows, so we route to it via gateway.
WIN_IP=$(ip route show | grep -m1 default | awk '{print $3}')

if [ -z "$WIN_IP" ]; then
    echo "⚠️  Could not detect Windows host IP, defaulting to localhost"
    GEMMA_URL="http://localhost:8001"
else
    echo "🖥️  Windows host IP: $WIN_IP"
    GEMMA_URL="http://${WIN_IP}:8001"
fi

# --- Check Gemma 4 server is reachable ---
echo "🔍  Checking Gemma 4 server at $GEMMA_URL …"
if curl -s --max-time 3 "$GEMMA_URL/health" > /dev/null 2>&1; then
    echo "✅  Gemma 4 server is reachable"
else
    echo "⚠️  WARNING: Cannot reach Gemma 4 server at $GEMMA_URL"
    echo "   Make sure server.py is running on Windows before placing a call."
fi

# --- Start the bridge ---
echo ""
echo "🚀  Starting AudioSocket bridge on port 9092…"
echo "    Gemma 4 URL : $GEMMA_URL"
echo "    Piper model : $(basename $PIPER_MODEL)"
echo "    Press Ctrl+C to stop."
echo ""

source "$SCRIPT_DIR/venv_bridge/bin/activate"

python3 "$BRIDGE_PY" \
    --gemma-url "$GEMMA_URL" \
    --piper-model "$PIPER_MODEL" \
    --port 9092 \
    "$@"   # pass any extra args (e.g., --debug)
