# Gemma Voice Assistant — System Architecture

> Three-tier architecture: **Voice Layer → AI Brain → API & Data**

---

## 📞 Voice Pipeline (Forward + Return)

```mermaid
flowchart LR
    %% ── OUTSIDE CALLER PATH ──
    UserA(["Outside\nCaller"]) -- "PSTN" --> Twilio["Twilio\nCloud"]
    Twilio -- "WSS" --> ngrok["ngrok\nTunnel"]
    ngrok --> TBridge["Twilio\nBridge"]
    TBridge -- "Audio Stream" --> Session

    %% ── LOCAL USER PATH ──
    LU(["Local\nUser"]) -- "SIP" --> Asterisk["Asterisk\nPBX"]
    Asterisk -- "AudioSocket" --> ABridge["Asterisk\nBridge"]
    ABridge -- "Audio Stream" --> Session

    %% ── AI BRAIN ──
    subgraph Brain["AI Brain"]
        Session["Session\nManager"]
        VAD["Silero\nVAD"]
        Whisper["Whisper\nSTT"]
        Gemma["Gemma-4 LLM\n(Receptionist · Assistant)"]
        Kokoro["Kokoro /\nPiper TTS"]

        Session -- "PCM Audio" --> VAD
        VAD -- "Speech\nSegments" --> Whisper
        Whisper -- "Text" --> Gemma
        Gemma -- "Response" --> Kokoro
    end

    %% ── TTS RETURN (dotted = return path) ──
    Kokoro -. "TTS Audio" .-> TBridge
    Kokoro -. "TTS Audio" .-> ABridge
```

---

## 🗄️ Data & Authentication Flow (Tool Calling)

```mermaid
flowchart LR
    %% ── LLM TOOL CALL CHAIN ──
    Gemma["Gemma-4 LLM\n(Receptionist Mode)"] -- "tool_call:\nquery_project_echo" --> Registry["Tool Executor\n(registry.py)"]
    Registry -- "get_local_user_token()" --> TokenStore[("local_token.txt")]
    Registry -- "Authenticated\nHTTP POST" --> QueryAPI

    subgraph PE["Project Echo Backend"]
        AuthAPI["POST\n/api/v1/auth/login"]
        QueryAPI["POST\n/api/v1/chat/query"]
        DB[("ChromaDB\nVector Store")]
        QueryAPI <-- "hybrid_query()" --> DB
    end

    QueryAPI -- "JSON Response\n(documents)" --> Registry
    Registry -- "Tool Result" --> Gemma

    %% ── AUTH FLOW ──
    LU(["Local User"]) -. "Login Request" .-> AuthAPI
    AuthAPI -- "JWT Token" --> TokenStore

    %% ── CONCEPTUAL ──
    UserA(["Outside Caller"]) -. "Answered in\nReceptionist Mode" .-> Gemma
```

---

## Key

| Line Style | Meaning |
|---|---|
| **Solid →** | Primary data flow (forward pipeline) |
| **Dotted -.->** | Return path / secondary flow (TTS audio, auth, identity) |

| Color | Component |
|---|---|
| 🟢 Green border | Users (Outside Caller, Local User) |
| 🔵 Blue fill | Voice infrastructure (Twilio, Asterisk, Bridges) |
| 🟣 Purple fill | AI processing (Session, Whisper, Gemma, Kokoro) |
| 🔴 Red fill | VAD gate / Databases (PostgreSQL, ChromaDB) |
| 🟠 Orange fill | Data layer (Tool Executor, APIs, Token) |

---

## System Capabilities

| | Capability | Description |
|---|---|---|
| 🎭 | **Dual Persona** | Receptionist & Assistant are modes of the same LLM, selected by caller identity |
| 🔇 | **Barge-in** | User can interrupt AI mid-speech — TTS playback cancels instantly on speech detection |
| 🔍 | **Identity Resolution** | Direct `asyncpg` queries to PostgreSQL (`linked_spaces` → `users`), runs in parallel during greeting |
| 📝 | **Post-Call Processing** | Async pipeline after hangup — Gemma summarizes transcript + sentiment analysis → stored to SQLite |
| 🌐 | **REST API** | FastAPI exposes `/v1/chat/completions`, `/chat-audio`, `/health` alongside voice bridges |
| 🛠️ | **Tool Calling** | LLM invokes registered tools (`query_project_echo`, `get_order_status`, `send_sms`) via `registry.py` |
