#!/usr/bin/env bash
# =============================================================================
# start_bridge.sh — Start the Asterisk AudioSocket Bridge in WSL2
# =============================================================================
# Run this script INSIDE WSL2 (not Windows PowerShell).
# Make executable: chmod +x start_bridge.sh
# Run: ./start_bridge.sh
#
# Prerequisites (see SETUP.md for full guide):
#   1. python3, pip, faster-whisper, gtts, httpx, pydub installed
#   2. server.py (Gemma 4) must already be running on Windows :8000
#   3. Asterisk must be running: sudo systemctl start asterisk
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE_PY="$SCRIPT_DIR/asterisk_bridge.py"

# --- Detect WSL2 Windows host IP (to reach server.py running on Windows) ---
# When running bridge inside WSL2, Windows is at the WSL2 gateway IP.
# The Gemma 4 server.py runs on Windows, so we route to it via gateway.
WIN_IP=$(ip route show | grep -m1 default | awk '{print $3}')

if [ -z "$WIN_IP" ]; then
    echo "⚠️  Could not detect Windows host IP, defaulting to localhost"
    GEMMA_URL="http://localhost:8000"
else
    echo "🖥️  Windows host IP: $WIN_IP"
    GEMMA_URL="http://${WIN_IP}:8000"
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
echo "    Gemma 4 URL: $GEMMA_URL"
echo "    Press Ctrl+C to stop."
echo ""

python3 "$BRIDGE_PY" \
    --gemma-url "$GEMMA_URL" \
    --port 9092 \
    "$@"   # pass any extra args (e.g., --debug)
