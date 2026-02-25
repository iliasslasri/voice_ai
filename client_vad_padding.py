import asyncio
import base64
import json
import os
import shutil
import subprocess

import websockets

# Configuration
SERVER_URL = "ws://localhost:8000/v1/realtime"
API_KEY = os.getenv("GRADIUM_API_KEY")
AUDIO_FILE_PATH = "record(10).wav"  # Your file
OUTPUT_FILENAME = "response.ogg"


def convert_wav_to_opus_ogg(
    file_path: str, save_path: str = "debug_output.ogg"
) -> bytes:
    """
    Converts WAV -> OGG/Opus AND adds 1.0s of silence to trigger the VAD.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    command = [
        "ffmpeg",
        "-y",
        "-i",
        file_path,
        # CRITICAL: Add exactly 1.0s of silence to the end.
        # This matches the server's silence_duration_ms setting below.
        "-af",
        "apad=pad_dur=1.0",
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
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ogg_bytes, stderr = process.communicate()

    print(f"💾 Saving debug audio to {save_path}...")
    with open(save_path, "wb") as f:
        f.write(ogg_bytes)

    if process.returncode != 0:
        print(f"❌ FFmpeg Error: {stderr.decode()}")
        return b""
    return ogg_bytes


async def run_client():
    if not API_KEY:
        print("❌ Error: GRADIUM_API_KEY is not set.")
        return

    # 1. Prepare Audio
    print(f"⚙️  Converting '{AUDIO_FILE_PATH}'...")
    ogg_data = convert_wav_to_opus_ogg(AUDIO_FILE_PATH)
    if not ogg_data:
        return
    print(f"   Ready to send {len(ogg_data)} bytes.")

    # 2. Connect
    headers = {"Authorization": f"Bearer {API_KEY}", "openai-beta": "realtime=v1"}
    print(f"🔌 Connecting to {SERVER_URL}...")

    async with websockets.connect(
        SERVER_URL, extra_headers=headers, subprotocols=["realtime"]
    ) as websocket:
        print("✅ Connected!")

        # 3. CONFIGURE VAD (The 'Magic Formula')
        # We tell the server: "If you hear silence for 500ms, reply."
        # Since we added 1000ms of silence to the file, this GUARANTEES a trigger.
        print("⚙️  Configuring Session...")
        vad_config = {
            "type": "session.update",
            "session": {
                "allow_recording": True,
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,  # Standard sensitivity
                    "prefix_padding_ms": 300,  # Keep start of speech
                    "silence_duration_ms": 500,  # <--- Shorter than our padding (1000ms)
                },
            },
        }
        await websocket.send(json.dumps(vad_config))
        await asyncio.sleep(0.2)

        # 4. Stream Audio
        print("🚀 Streaming audio...")
        CHUNK_SIZE = 4096
        for i in range(0, len(ogg_data), CHUNK_SIZE):
            chunk = ogg_data[i : i + CHUNK_SIZE]
            event = {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode("utf-8"),
            }
            await websocket.send(json.dumps(event))
            await asyncio.sleep(0.05)

        print("✅ Audio sent! Waiting for VAD trigger...")

        # 5. Start Player
        if shutil.which("ffplay"):
            player_process = subprocess.Popen(
                ["ffplay", "-autoexit", "-nodisp", "-"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            player_process = None

        # 6. Listen Loop
        while True:
            try:
                # Wait up to 30s because the AI needs time to think/transcribe
                msg = await asyncio.wait_for(websocket.recv(), timeout=30.0)
            except asyncio.TimeoutError:
                print("❌ Timeout.")
                break

            data = json.loads(msg)
            evt = data.get("type")

            if evt == "response.audio.delta":
                b64 = data.get("delta")
                if b64:
                    chunk = base64.b64decode(b64)
                    with open(OUTPUT_FILENAME, "ab") as f:
                        f.write(chunk)
                    if player_process:
                        try:
                            player_process.stdin.write(chunk)
                            player_process.stdin.flush()
                            print(".", end="", flush=True)
                        except:
                            break

            elif evt == "response.text.delta":
                print(f"\n🤖 AI: {data.get('delta')}")

            # Debug: See when the server detects speech start/end
            elif evt == "input_audio_buffer.speech_started":
                print("\n🎙️  [Server Hearing Speech]")
            elif evt == "input_audio_buffer.speech_stopped":
                print("\n🛑 [Server Detected Silence - Processing]")

            elif evt == "response.done":
                print("\n✅ Response finished.")
                break
            elif evt == "error":
                print(f"\n❌ Server Error: {data}")
                break

        if player_process:
            player_process.stdin.close()
            player_process.wait()


if __name__ == "__main__":
    if os.path.exists(OUTPUT_FILENAME):
        os.remove(OUTPUT_FILENAME)
    asyncio.run(run_client())
