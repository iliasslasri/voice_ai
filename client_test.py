import asyncio
import base64
import io
import json
import os
import shutil
import subprocess
import time

import numpy as np
import websockets
from gtts import gTTS

# Configuration
SERVER_URL = "ws://localhost:8000/v1/realtime"
API_KEY = os.getenv("GRADIUM_API_KEY")
AUDIO_FILE_PATH = "john_reference.wav"
AUDIO_FILE_PATH = "cocktail_party_sample.wav"
OUTPUT_FILENAME = "temp_output.ogg"


def convert_wav_to_opus_ogg(file_path: str) -> bytes:
    """Reads a WAV file and converts it to OGG/Opus bytes using FFmpeg."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # FFmpeg command to read file -> resample to 24k -> mono -> encode Opus -> stdout
    command = [
        "ffmpeg",
        "-i",
        file_path,  # Input file
        "-af",
        "apad=pad_dur=2.0",
        "-ar",
        "24000",  # Resample to 24kHz (Unmute standard)
        "-ac",
        "1",  # Mono
        "-c:a",
        "libopus",  # Encode to Opus
        "-b:a",
        "24k",  # Bitrate
        "-f",
        "ogg",  # Output container
        "pipe:1",  # Write to stdout
    ]

    # Run FFmpeg
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ogg_bytes, stderr = process.communicate()

    if process.returncode != 0:
        print(f"❌ FFmpeg Error: {stderr.decode()}")
        return b""

    return ogg_bytes


async def run_client():
    if not API_KEY:
        print("❌ Error: GRADIUM_API_KEY is not set.")
        return

    # --- STEP 1: Prepare Audio ---
    print(f"Converting '{AUDIO_FILE_PATH}' to OGG/Opus...")
    try:
        ogg_data = convert_wav_to_opus_ogg(AUDIO_FILE_PATH)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    if not ogg_data:
        return
    print(f"   Ready to send {len(ogg_data)} bytes of audio.")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "openai-beta": "realtime=v1",
    }

    print(f"🔌 Connecting to {SERVER_URL}...")
    try:
        async with websockets.connect(
            SERVER_URL, extra_headers=headers, subprotocols=["realtime"]
        ) as websocket:

            print("Connected!")

            # Swallow handshake if present
            try:
                await asyncio.wait_for(websocket.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                pass

            print("Streaming audio...")

            # --- STEP 3: Send Audio Chunks ---
            CHUNK_SIZE = 1920
            for i in range(0, len(ogg_data), CHUNK_SIZE):
                chunk = ogg_data[i : i + CHUNK_SIZE]

                # Wrap in JSON event
                event = {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("utf-8"),
                }

                await websocket.send(json.dumps(event))
                await asyncio.sleep(0.05)  # Prevent flooding

            # --- Commit ---
            # Tell the AI we are done talking so it processes the audio
            # await websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))
            print("Audio sent! Listening for response...")
            if not shutil.which("ffplay"):
                print("⚠️ 'ffplay' not found. Cannot play audio. Install ffmpeg.")
                player_process = None
            else:
                player_process = subprocess.Popen(
                    ["ffplay", "-autoexit", "-nodisp", "-"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            # --- Listen for AI Response ---
            while True:
                msg = await websocket.recv()
                data = json.loads(msg)
                evt = data.get("type")
                if evt == "response.audio.delta":
                    print("[AI Audio Stream...]")
                    # 1. Get Base64 string
                    b64_string = data.get("delta")

                    if b64_string:
                        # Decode to binary Opus/OGG bytes
                        audio_bytes = base64.b64decode(b64_string)

                        # Append directly to file (we accumulate the stream here)
                        with open(OUTPUT_FILENAME, "ab") as f:
                            f.write(audio_bytes)

                        # --- STREAM LOGIC ---
                        # Write the chunk directly to the player's input pipe
                        try:
                            player_process.stdin.write(audio_bytes)
                            player_process.stdin.flush()  # Force play immediately
                            print(".", end="", flush=True)
                        except BrokenPipeError:
                            print("\n❌ Player closed unexpectedly.")
                            break
                elif evt == "response.text.delta":
                    print(f"🤖 AI: {data.get('delta')}")
                elif evt == "response.done":
                    print("Response finished.")
                    break
                elif evt == "error":
                    print(f"Server Error: {data}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    if os.path.exists(OUTPUT_FILENAME):
        os.remove(OUTPUT_FILENAME)
    asyncio.run(run_client())
