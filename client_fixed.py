import asyncio
import base64
import io
import json
import os
import subprocess
import time

import websockets
from gtts import gTTS

# Configuration
SERVER_URL = "ws://localhost:8000/v1/realtime"
API_KEY = os.getenv("GRADIUM_API_KEY")
OUTPUT_FILENAME = "output.ogg"


def generate_speech_bytes(text: str) -> bytes:
    """Generates speech OGG/Opus from text."""
    print(f"🗣️  Generating voice for: '{text}'...")
    tts = gTTS(text)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)

    # Convert MP3 to OGG/Opus WITH SILENCE PADDING
    # We add '-af apad=pad_dur=2.0' to add 2 seconds of silence at the end.
    # This replaces the need for a 'commit' message.
    command = [
        "ffmpeg",
        "-f",
        "mp3",
        "-i",
        "pipe:0",
        "-ar",
        "24000",
        "-ac",
        "1",
        "-c:a",
        "libopus",
        "-b:a",
        "24k",
        "-f",
        "ogg",
        "pipe:1",
        "-af apad=pad_dur=2.0",
    ]
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    ogg_bytes, stderr = process.communicate(input=fp.read())

    if process.returncode != 0:
        print(f"❌ FFmpeg Error: {stderr.decode()}")
        return b""

    return ogg_bytes


async def run_client():
    if not API_KEY:
        print("❌ Error: GRADIUM_API_KEY is not set.")
        return

    # Clean up previous run
    if os.path.exists(OUTPUT_FILENAME):
        os.remove(OUTPUT_FILENAME)

    # --- STEP 1: Prepare Audio ---
    # This now includes the silence inside the bytes!
    speech_ogg = generate_speech_bytes("Hello! Can you hear me?")

    if not speech_ogg:
        return
    print(f"🚀 Prepared {len(speech_ogg)} bytes (Speech + Silence)...")

    # --- STEP 2: Connect ---
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "openai-beta": "realtime=v1",
    }

    print(f"🔌 Connecting to {SERVER_URL}...")
    try:
        async with websockets.connect(
            SERVER_URL, extra_headers=headers, subprotocols=["realtime"]
        ) as websocket:

            print("✅ Connected!")

            # Swallow handshake if present
            try:
                await asyncio.wait_for(websocket.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                pass

            print("🚀 Streaming audio...")
            chunk_size = 4096

            # --- STEP 3: Send Audio Chunks ---
            for i in range(0, len(speech_ogg), chunk_size):
                chunk = speech_ogg[i : i + chunk_size]
                event = {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("utf-8"),
                }
                await websocket.send(json.dumps(event))
                await asyncio.sleep(0.05)

            # --- REMOVED STEP 4 (Commit) ---
            # We do NOT send 'commit'. The 2s of silence in the audio stream
            # will automatically tell the server we are done.
            print("✅ Audio sent! Waiting for VAD trigger...")

            # --- STEP 5: Listen for AI Response ---
            while True:
                msg = await websocket.recv()
                data = json.loads(msg)
                evt = data.get("type")

                if evt == "response.audio.delta":
                    b64_string = data["delta"]
                    audio_bytes = base64.b64decode(b64_string)
                    with open(OUTPUT_FILENAME, "ab") as f:
                        f.write(audio_bytes)
                    print(".", end="", flush=True)

                elif evt == "response.text.delta":
                    print(f"\n🤖 AI: {data.get('delta')}")

                elif evt == "response.done":
                    print("\n✅ Response finished.")
                    break

                elif evt == "error":
                    print(f"\n❌ Server Error: {data}")
                    break

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(run_client())
