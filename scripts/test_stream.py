import asyncio
import websockets
import json
import base64
import sounddevice as sd
import numpy as np
import audioop

# Configuration
WS_URL = "ws://localhost:8000/twiliostream"
RATE = 8000

async def mic_to_ws(ws):
    print("[*] Microphone active. Speak now!")
    def callback(indata, frames, time, status):
        linear_data = (indata * 32767).astype(np.int16).tobytes()
        mulaw_data = audioop.lin2ulaw(linear_data, 2)
        payload = {
            "event": "media",
            "media": {"payload": base64.b64encode(mulaw_data).decode('utf-8')}
        }
        loop.call_soon_threadsafe(asyncio.create_task, ws.send(json.dumps(payload)))

    with sd.InputStream(samplerate=RATE, channels=1, callback=callback):
        while True:
            await asyncio.sleep(1)

async def ws_to_speaker(ws):
    stream = sd.OutputStream(samplerate=RATE, channels=1)
    stream.start()
    print("[*] Speaker active. Waiting for AI...")
    while True:
        message = await ws.recv()
        data = json.loads(message)
        if data['event'] == 'media':
            mulaw_data = base64.b64decode(data['media']['payload'])
            linear_data = audioop.ulaw2lin(mulaw_data, 2)
            audio_array = np.frombuffer(linear_data, dtype=np.int16).astype(np.float32) / 32767.0
            stream.write(audio_array)

async def main():
    global loop
    loop = asyncio.get_running_loop()
    async with websockets.connect(WS_URL) as ws:
        # Start session
        await ws.send(json.dumps({"event": "start", "start": {"callSid": "test-call", "streamSid": "test-stream"}}))
        await asyncio.gather(mic_to_ws(ws), ws_to_speaker(ws))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[*] Stopped.")
