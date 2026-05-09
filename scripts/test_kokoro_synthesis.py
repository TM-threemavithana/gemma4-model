import os
import wave
import numpy as np
from src.adapters.kokoro import KokoroAdapter

def test_synthesis():
    adapter = KokoroAdapter()
    print("Loading Kokoro...")
    adapter.load()
    
    text = "Hello, I am now using the Kokoro text to speech engine. This should sound much more natural."
    print(f"Synthesizing: '{text}'")
    pcm_data = adapter.synthesize_to_pcm8k(text)
    
    print(f"Received {len(pcm_data)} bytes of PCM data")
    
    # Save to a wav file to verify sample rate
    output_path = "output/test_kokoro_8k.wav"
    os.makedirs("output", exist_ok=True)
    
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(pcm_data)
        
    print(f"Saved to {output_path}")
    print(f"PCM length: {len(pcm_data)/16000:.2f} seconds at 8kHz")

if __name__ == "__main__":
    try:
        test_synthesis()
    except Exception as e:
        print(f"Test failed: {e}")
