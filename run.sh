#!/bin/bash
# ============================================================
# run.sh — One-command launcher for the Gemma 4 server
# ============================================================
# Usage:
#   ./run.sh                          # CPU backend, port 8000
#   ./run.sh --backend gpu            # GPU backend
#   ./run.sh --port 9000              # custom port
#   ./run.sh --backend gpu --port 9000
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Colors for output ----
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${CYAN}╔══════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║     🚀 Gemma 4 Server Launcher           ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════╝${NC}"
echo ""

# ---- Check / create virtual environment ----
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}📦 Creating virtual environment...${NC}"
    python3 -m venv venv
fi

echo -e "${GREEN}🔌 Activating virtual environment...${NC}"
source venv/bin/activate

# ---- Install / upgrade deps ----
DEPS="fastapi uvicorn litert-lm-api-nightly"
echo -e "${GREEN}📥 Checking dependencies...${NC}"
pip install --quiet --upgrade $DEPS 2>/dev/null

# ---- Locate model ----
MODEL_PATH="${GEMMA_MODEL_PATH:-$SCRIPT_DIR/gemma-4-E2B-it.litertlm}"
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${YELLOW}⚠️  Model not found at: $MODEL_PATH${NC}"
    echo "Set GEMMA_MODEL_PATH or place model in $SCRIPT_DIR"
    exit 1
fi

echo -e "${GREEN}✅ Model found: $(basename $MODEL_PATH)${NC}"
echo ""

# ---- Launch server ----
echo -e "${CYAN}🌐 Starting server...${NC}"
echo -e "${CYAN}   Press Ctrl+C to stop${NC}"
echo ""

python server.py --model "$MODEL_PATH" "$@"
