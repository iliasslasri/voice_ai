import asyncio
import base64
import json
import os
import subprocess

import websockets

# Configuration
SERVER_URL = "ws://localhost:8000/v1/realtime"
API_KEY = os.getenv("GRADIUM_API_KEY")

# Your specific file path
AUDIO_FILE_PATH = "/home/iliass/voice_ai/cocktail_party_sample.wav"


def convert_wav_to_opus_ogg(file_path: str) -> bytes:
    """Reads a WAV file and converts it to OGG/Opus bytes using FFmpeg."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # FFmpeg command to read file -> resample to 24k -> mono -> encode Opus -> stdout
    command = [
        "ffmpeg",
        "-i",
        file_path,  # Input file
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
    print(f"⚙️  Converting '{AUDIO_FILE_PATH}' to OGG/Opus...")
    try:
        ogg_data = convert_wav_to_opus_ogg(AUDIO_FILE_PATH)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    if not ogg_data:
        return
    print(f"   Ready to send {len(ogg_data)} bytes of audio.")

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

            # --- STEP 3: Send Audio Chunks ---
            CHUNK_SIZE = 4096  # Send 4KB at a time
            for i in range(0, len(ogg_data), CHUNK_SIZE):
                chunk = ogg_data[i : i + CHUNK_SIZE]

                # Wrap in JSON event
                event = {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("utf-8"),
                }

                await websocket.send(json.dumps(event))
                await asyncio.sleep(0.05)  # Prevent flooding

            # --- STEP 4: Commit ---
            # Tell the AI we are done talking so it processes the audio
            await websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))
            print("✅ Audio sent! Listening for response...")

            # --- STEP 5: Listen for AI Response ---
            while True:
                msg = await websocket.recv()
                data = json.loads(msg)
                evt = data.get("type")
                if evt == "response.audio.delta":
                    print("🔊 [AI Audio Stream...]")
                elif evt == "response.text.delta":
                    print(f"🤖 AI: {data.get('delta')}")
                elif evt == "response.done":
                    print("✅ Response finished.")
                    break
                elif evt == "error":
                    print(f"❌ Server Error: {data}")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(run_client())
