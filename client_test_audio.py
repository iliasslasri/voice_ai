import asyncio
import base64
import json
import os
import queue
import shutil
import subprocess
import sys

import numpy as np
import sounddevice as sd
import websockets

# --- Configuration ---
SERVER_URL = "ws://localhost:8000/v1/realtime"
API_KEY = os.getenv("GRADIUM_API_KEY")

# --- Audio Settings ---
# We capture at 48kHz (Standard Mic) and send at 24kHz (Server Requirement)
MIC_SAMPLE_RATE = 48000
SERVER_SAMPLE_RATE = 24000
CHANNELS = 1
DTYPE = "int16"
CHUNK_DURATION_MS = 100
MIC_BLOCK_SIZE = int(MIC_SAMPLE_RATE * (CHUNK_DURATION_MS / 1000))

audio_queue = queue.Queue()


def audio_callback(indata, frames, time, status):
    """
    Captures 48kHz audio, downsamples to 24kHz, and queues it.
    """
    if status:
        pass  # Ignore overflows for smooth streaming

    # 1. Downsample (48k -> 24k) by taking every 2nd sample
    downsampled = indata[::2, 0]

    # 2. Ensure Int16
    clean_data = np.ascontiguousarray(downsampled, dtype=np.int16)
    audio_queue.put(clean_data)


async def send_mic_audio(websocket):
    """Continuously sends mic audio to server."""
    print(f"🎙️  Microphone active! (Streaming {SERVER_SAMPLE_RATE}Hz PCM)")

    while True:
        try:
            while audio_queue.empty():
                await asyncio.sleep(0.005)

            data = audio_queue.get()

            # Convert Int16 -> Bytes -> Base64
            pcm_bytes = data.tobytes()
            b64_audio = base64.b64encode(pcm_bytes).decode("utf-8")

            event = {"type": "input_audio_buffer.append", "audio": b64_audio}
            await websocket.send(json.dumps(event))

        except Exception as e:
            print(f"❌ Send Error: {e}")
            break


async def receive_and_play(websocket):
    """Receives AI audio and plays it via ffplay."""
    if not shutil.which("ffplay"):
        print("❌ Error: 'ffplay' not found.")
        return

    # Start Player (Standard Input -> Speakers)
    player_process = subprocess.Popen(
        ["ffplay", "-autoexit", "-nodisp", "-f", "ogg", "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    print("🔊 Speaker ready.")

    try:
        while True:
            msg = await websocket.recv()
            data = json.loads(msg)
            evt = data.get("type")

            if evt == "response.audio.delta":
                b64_string = data.get("delta")
                if b64_string:
                    try:
                        audio_bytes = base64.b64decode(b64_string)
                        player_process.stdin.write(audio_bytes)
                        player_process.stdin.flush()
                        print(".", end="", flush=True)
                    except BrokenPipeError:
                        break

            elif evt == "response.text.delta":
                print(f"\n🤖 AI: {data.get('delta')}")

            elif evt == "input_audio_buffer.speech_started":
                print("\n[Speech Detected]", end="", flush=True)

            elif evt == "input_audio_buffer.speech_stopped":
                print("\n[Silence - Processing]", end="", flush=True)

            elif evt == "error":
                print(f"\n❌ Server Error: {data}")

    except Exception as e:
        print(f"\n❌ Receiver Error: {e}")
    finally:
        if player_process:
            player_process.stdin.close()
            player_process.wait()


async def run_client():
    if not API_KEY:
        print("❌ Error: GRADIUM_API_KEY is not set.")
        return

    headers = {"Authorization": f"Bearer {API_KEY}", "openai-beta": "realtime=v1"}

    print(f"🔌 Connecting to {SERVER_URL}...")
    async with websockets.connect(
        SERVER_URL, extra_headers=headers, subprotocols=["realtime"]
    ) as websocket:
        print("✅ Connected!")

        # --- STEP 1: CONFIGURATION (THE FIX) ---
        # We added 'allow_recording': True to fix the validation error.
        print("⚙️  Configuring Session...")
        setup_msg = {
            "type": "session.update",
            "session": {
                "input_audio_format": "pcm16",
                "allow_recording": True,  # <--- CRITICAL FIX
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "silence_duration_ms": 600,
                },
            },
        }
        await websocket.send(json.dumps(setup_msg))

        # Wait for settings to apply
        await asyncio.sleep(0.5)

        # --- STEP 2: START LOOPS ---
        # Note: We open Mic at 48000Hz (Hardware) but Callback creates 24000Hz (Server)
        with sd.InputStream(
            samplerate=MIC_SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=MIC_BLOCK_SIZE,
            callback=audio_callback,
        ):

            send_task = asyncio.create_task(send_mic_audio(websocket))
            receive_task = asyncio.create_task(receive_and_play(websocket))

            done, pending = await asyncio.wait(
                [send_task, receive_task], return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(run_client())
    except KeyboardInterrupt:
        print("\n👋 Stopped by user.")
