import asyncio
import base64
import json
import os
import shutil
import subprocess
import uuid

import websockets

# --- Configuration ---
SERVER_URL = "ws://localhost:8000/v1/realtime"
API_KEY = os.getenv("GRADIUM_API_KEY", "dummy-key")

# Input/Output
INPUT_AUDIO_PATH = "cocktail_party_sample.wav"
# INPUT_AUDIO_PATH = "Iliass.wav"
OUTPUT_AUDIO_PATH = "response_audio.ogg"

# Check for ffplay
HAS_FFPLAY = shutil.which("ffplay") is not None


def stream_audio_chunks(file_path: str, chunk_size: int = 4096):
    """
    Streams audio directly from FFmpeg as bytes (Generator).
    Matches the input requirements: 24kHz, Mono, OGG/Opus.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    command = [
        "ffmpeg",
        "-i",
        file_path,
        "-af",
        "apad=pad_dur=1.5",  # VAD padding
        "-c:a",
        "libopus",
        "-b:a",
        "24k",
        "-ar",
        "24000",
        "-ac",
        "1",
        "-f",
        "ogg",
        "-v",
        "error",
        "pipe:1",  # Output to stdout
    ]

    # Open process with piped stdout
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )

    try:
        while True:
            # Read exact chunk size from the pipe
            data = process.stdout.read(chunk_size)
            if not data:
                break
            yield data
    finally:
        # Cleanup
        process.stdout.close()
        process.wait()


def base64_encode_opus(chunk: bytes) -> str:
    """
    Equivalent to the JS logic: base64EncodeOpus(opus)
    """
    return base64.b64encode(chunk).decode("utf-8")


def create_event(event_type: str, data: dict = None):
    event = {
        "event_id": str(uuid.uuid4()),
        "type": event_type,
    }
    if data:
        event.update(data)
    return event


async def get_audio_response():
    # Clean previous output
    if os.path.exists(OUTPUT_AUDIO_PATH):
        os.remove(OUTPUT_AUDIO_PATH)

    headers = {"Authorization": f"Bearer {API_KEY}", "openai-beta": "realtime=v1"}

    print(f"Connecting to {SERVER_URL}...")
    try:
        async with websockets.connect(
            SERVER_URL, extra_headers=headers, subprotocols=["realtime"]
        ) as websocket:
            print("Connected.")

            # --- Configure Session ---
            session_config = {
                "session": {
                    "allow_recording": True,
                    "instructions": {
                        "type": "constant",
                        "content": "You are a helpful assistant. Answer briefly.",
                    },
                }
            }
            await websocket.send(
                json.dumps(create_event("session.update", session_config))
            )
            print("Session configured.")

            # --- Stream Audio ---
            print(f"🎤 Streaming audio from {INPUT_AUDIO_PATH}...")

            for opus_chunk in stream_audio_chunks(INPUT_AUDIO_PATH):
                event = create_event(
                    "input_audio_buffer.append",
                    {"audio": base64_encode_opus(opus_chunk)},
                )
                await websocket.send(json.dumps(event))
                await asyncio.sleep(0.01)

            # await websocket.send(json.dumps(create_event("input_audio_buffer.commit")))

            print("Audio stream finished. Waiting for response...")

            # --- Handle Response ---
            if HAS_FFPLAY:
                player_process = subprocess.Popen(
                    ["ffplay", "-autoexit", "-nodisp", "-"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                player_process = None

            while True:
                msg = await websocket.recv()
                data = json.loads(msg)
                evt_type = data.get("type")

                if evt_type == "response.audio.delta":
                    b64_string = data.get("delta")
                    if b64_string:
                        audio_bytes = base64.b64decode(b64_string)
                        with open(OUTPUT_AUDIO_PATH, "ab") as f:
                            f.write(audio_bytes)
                        if player_process:
                            try:
                                player_process.stdin.write(audio_bytes)
                                player_process.stdin.flush()
                            except BrokenPipeError:
                                pass
                        print(".", end="", flush=True)

                elif evt_type == "response.text.delta":
                    print(f"\nAI: {data.get('delta')}", end="")

                elif evt_type == "conversation.item.input_audio_transcription.delta":
                    print(f"\nYou: {data.get('delta')}", end="")

                elif evt_type == "error":
                    error_details = data.get("error", {})
                    # Check if it's just a warning (like cold starts)
                    if error_details.get("type") == "warning":
                        print(
                            f"\nServer Warning: {error_details.get('message')} (Waiting...)"
                        )
                        continue  # Keep listening! Don't break.

                    # Real error -> Stop
                    print(f"\n❌ Server Error: {data}")
                    break

                elif evt_type == "response.done":
                    print("\n✅ Response complete.")
                    break

            if player_process:
                player_process.stdin.close()
                player_process.wait()

    except Exception as e:
        print(f"\n❌ Error: {e}")

    if os.path.exists(OUTPUT_AUDIO_PATH):
        print(f"\n💾 Response saved to: {OUTPUT_AUDIO_PATH}")


if __name__ == "__main__":
    asyncio.run(get_audio_response())
