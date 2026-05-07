# Gemma 4 AI Calling Agent — Complete Setup Guide
## Asterisk + WSL2 + Free SIP Softphone (Linphone)

> **Goal:** Call a phone number (or SIP extension), speak to Gemma 4, and hear it respond — all locally on your Windows machine using WSL2.

---

## Architecture Recap

```
📱 Linphone App (your phone/PC)
       │  SIP  (port 5060)
       ▼
┌─────────────────────────┐
│  Asterisk PBX (WSL2)    │   ← Linux telephony engine
│  extensions.conf        │
└──────────┬──────────────┘
           │  AudioSocket TCP (port 9092)  raw PCM audio
           ▼
┌─────────────────────────┐
│  asterisk_bridge.py     │   ← Python bridge (WSL2)
│  • VAD silence detect   │
│  • faster-whisper STT   │
│  • Gemma 4 via HTTP     │
│  • gTTS → PCM           │
└──────────┬──────────────┘
           │  HTTP (port 8000)
           ▼
┌─────────────────────────┐
│  server.py (Windows)    │   ← Already running, no changes
│  Gemma 4 + FastAPI      │
└─────────────────────────┘
```

---

## Step 1 — Enable WSL2 (if not already done)

Open **PowerShell as Administrator** and run:

```powershell
wsl --install
# Restart your PC when prompted
# After restart, Ubuntu will open and ask for a username/password
```

If WSL is already installed, verify it's version 2:
```powershell
wsl --list --verbose
# Should show VERSION 2 next to Ubuntu
wsl --set-version Ubuntu 2    # upgrade if it shows 1
```

---

## Step 2 — Install Asterisk in WSL2

Open WSL2 terminal (search "Ubuntu" in Start menu) and run:

```bash
# Update packages
sudo apt update && sudo apt upgrade -y

# Install Asterisk
sudo apt install -y asterisk

# Verify it's installed
asterisk --version
# Expected: Asterisk 18.x.x or 20.x.x
```

---

## Step 3 — Configure Asterisk

```bash
# Back up defaults
sudo cp /etc/asterisk/extensions.conf /etc/asterisk/extensions.conf.bak
sudo cp /etc/asterisk/sip.conf        /etc/asterisk/sip.conf.bak
sudo cp /etc/asterisk/modules.conf    /etc/asterisk/modules.conf.bak

# Access Windows files from WSL2 (your project folder):
WIN_PROJECT="/mnt/c/Users/User/gemma4-model"

# Copy the config templates from your project:
sudo cp "$WIN_PROJECT/asterisk/extensions.conf" /etc/asterisk/extensions.conf
sudo cp "$WIN_PROJECT/asterisk/sip.conf"        /etc/asterisk/sip.conf
sudo cp "$WIN_PROJECT/asterisk/modules.conf"    /etc/asterisk/modules.conf

# Verify AudioSocket module exists:
ls /usr/lib/asterisk/modules/app_audiosocket.so
# If NOT found, install:
sudo apt install -y asterisk-modules
```

---

## Step 4 — Start Asterisk

```bash
# Start Asterisk service
sudo systemctl start asterisk

# Enable auto-start on WSL2 boot:
sudo systemctl enable asterisk

# Check status:
sudo systemctl status asterisk

# Open Asterisk console (useful for debugging):
sudo asterisk -rvvvvv

# Inside the Asterisk console, verify AudioSocket module is loaded:
# asterisk*CLI> module show like audiosocket
# Should show: app_audiosocket.so
```

---

## Step 5 — Install Python Dependencies in WSL2

```bash
cd /mnt/c/Users/User/gemma4-model

# Create a virtual environment (recommended)
python3 -m venv venv_bridge
source venv_bridge/bin/activate

# Install bridge dependencies
pip install -r requirements_calling.txt

# If you hit "audioop not found" on Python 3.13+:
pip install audioop-lts
```

---

## Step 6 — Find Your WSL2 IP Address

The SIP softphone (Linphone) needs to know where Asterisk is running.

```bash
# In WSL2 terminal:
hostname -I
# Example output: 172.20.xxx.xxx
# This is your WSL2 IP — note it down!
```

> **Note:** WSL2 IP changes on reboot. For a static IP, see the "Static IP" section below.

---

## Step 7 — Install Linphone (Free SIP Softphone)

**Option A: Linphone on Android/iPhone**
1. Search "Linphone" in Google Play / App Store — install (free)
2. Open → **Assistant** → **Use SIP account**
3. Enter:
   - **Username**: `gemmaphone`
   - **Password**: `gemma1234`
   - **Domain**: `<your WSL2 IP>` (from Step 6)
   - **Transport**: UDP
4. Tap **Login** — you should see a green dot (registered ✅)

**Option B: Linphone Desktop (PC)**
1. Download from https://www.linphone.org/technical-corner/linphone
2. Open → **New SIP Account** → enter same credentials as above

---

## Step 8 — Start Everything

### Terminal 1 (WSL2): Start Gemma 4 Server
```bash
# Start your existing Gemma server inside WSL (where the model is located)
cd ~/gemma-server
./run.sh
# Should say: [OK] Server ready on 0.0.0.0:8000
```

### Terminal 2 (WSL2): Start the AudioSocket Bridge
```bash
cd /mnt/c/Users/User/gemma4-model
source venv_bridge/bin/activate
./start_bridge.sh
# Or manually:
# python3 asterisk_bridge.py --gemma-url http://<WINDOWS_IP>:8000

# Expected output:
# ⏳  Pre-loading Whisper model…
# ✅  Whisper loaded
# ✅  AudioSocket bridge listening on 0.0.0.0:9092
```

### Terminal 3 (WSL2): Verify Asterisk is running
```bash
sudo asterisk -rx "core show channels"
# Should say: 0 active channels
sudo asterisk -rx "module show like audiosocket"
# Should list app_audiosocket.so
```

---

## Step 9 — Make Your First AI Call! 📞

1. Open Linphone on your phone/PC
2. Make sure it shows **Registered** (green dot)
3. Dial **1000**
4. You should hear: *"Hello! I'm Gemma, your AI assistant. How can I help you today?"*
5. Wait for the beep/silence, then speak
6. Gemma will respond with synthesized voice!

---

## Testing Without a Phone (Quick Test)

Before setting up Linphone, test the bridge alone:

```bash
# Terminal 1: Start server.py (Windows)
# Terminal 2 (WSL2): Start bridge
cd /mnt/c/Users/User/gemma4-model
source venv_bridge/bin/activate
python3 asterisk_bridge.py &

# Terminal 3 (WSL2): Run the test client
python3 test_bridge.py --text "Hello Gemma, what is the capital of France?"
# → Saves response to bridge_response.wav
# → Play: aplay bridge_response.wav
```

---

## Troubleshooting

### "AudioSocket module not found"
```bash
sudo apt install -y asterisk asterisk-modules
sudo asterisk -rx "module load app_audiosocket.so"
```

### "SIP phone not registering"
```bash
# Check Asterisk is listening on port 5060:
sudo ss -ulnp | grep 5060

# Check firewall (WSL2 usually has no firewall, but just in case):
sudo ufw allow 5060/udp
sudo ufw allow 9092/tcp

# Reload SIP:
sudo asterisk -rx "sip reload"
sudo asterisk -rx "sip show peers"
# Should list "gemmaphone" with status
```

### "Gemma 4 unreachable from WSL2"
```bash
# Find Windows IP from WSL2:
ip route show | grep default | awk '{print $3}'

# Test connectivity:
curl http://<WINDOWS_IP>:8000/health

# Make sure Windows Firewall allows port 8000 from WSL2:
# In Windows: Settings → Windows Security → Firewall → Advanced → Inbound Rules
# Add rule: Allow TCP port 8000 from 172.16.0.0/12 (WSL2 range)
```

### "Can't hear audio / audio choppy"
```bash
# Check the bridge logs for TTS errors
python3 asterisk_bridge.py --debug 2>&1 | head -50

# Make sure pydub + ffmpeg is installed:
sudo apt install -y ffmpeg
pip install pydub
```

### "Call connects but no response from Gemma"
```bash
# Check server.py is running and healthy:
curl http://localhost:8000/health

# Check bridge is connected to correct Gemma URL:
./start_bridge.sh --debug
# Look for "Asking Gemma 4…" and "Gemma: <response>" in logs
```

---

## Optional: Static WSL2 IP (so it doesn't change on reboot)

In WSL2, create `/etc/wsl.conf`:
```bash
sudo tee /etc/wsl.conf << 'EOF'
[network]
generateHosts = false
generateResolvConf = false
EOF
```

Then in Windows PowerShell (as Admin):
```powershell
# Assign static IP to WSL2 interface (run at login or in Task Scheduler)
wsl -e sudo ip addr add 192.168.50.10/24 dev eth0
```

---

## File Reference

| File | Purpose |
|------|---------|
| `asterisk_bridge.py` | Main bridge — AudioSocket ↔ STT ↔ Gemma 4 ↔ TTS |
| `test_bridge.py` | Test the bridge without Asterisk or a phone |
| `start_bridge.sh` | Easy startup script for WSL2 |
| `requirements_calling.txt` | Python dependencies for the bridge |
| `asterisk/extensions.conf` | Asterisk dialplan (copy to `/etc/asterisk/`) |
| `asterisk/sip.conf` | SIP registration config (copy to `/etc/asterisk/`) |
| `asterisk/modules.conf` | Asterisk module loading (copy to `/etc/asterisk/`) |
| `server.py` | Your existing Gemma 4 server — **not modified** |

---

## Tuning the Voice Agent

Edit `asterisk_bridge.py` top-level constants:

```python
DEFAULT_SYSTEM_MSG = "..."     # Change Gemma's persona/instructions
SILENCE_THRESHOLD_DB = 500     # Higher = more sensitive to silence
SILENCE_FRAMES_NEEDED = 40     # Frames of silence before processing (40 = 0.8s)
MIN_SPEECH_FRAMES = 10         # Minimum frames to bother transcribing
```

Or pass via CLI:
```bash
python3 asterisk_bridge.py \
  --silence-threshold 300 \
  --system-msg "You are a helpful bank assistant. Be very concise." \
  --debug
```
