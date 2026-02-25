import queue
import threading
import time

import numpy as np
import sounddevice as sd
import torch
from pyannote.audio import Pipeline

HF_TOKEN = "YOUR_HUGGINGFACE_TOKEN"
SAMPLE_RATE = 16000
CHUNK_DURATION = 2.0
overlap_duration = 0.5
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)


class CocktailDeMixer:
    def __init__(self):
        print("Loading Pyannote Pipeline (this takes a moment)...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.diarization = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", use_auth_token=True
            ).to(self.device)
        except Exception as e:
            print(f"Error loading Pyannote: {e}")
            exit()

        print(f"Pipeline ready on {self.device}")
        self.target_speaker = None

    def get_diarization(self, audio_buffer):
        """
        Runs Pyannote on the buffer to find who is speaking when.
        """
        # Pyannote expects a Tensor (channels, time)
        audio_tensor = torch.tensor(audio_buffer).float().unsqueeze(0).to(self.device)

        # We wrap this in a customized 'file' object structure or pass directly if supported
        # For the hackathon, we use the raw inference wrapper
        # Note: In a full app, you'd wrap this properly for the Pipeline input
        diarization = self.diarization(
            {"waveform": audio_tensor, "sample_rate": SAMPLE_RATE}
        )
        return diarization

    def repair_segment(self, audio_chunk, overlap_mask):
        """
        THIS IS THE MAGIC SAUCE (Moshi / Gradium Integration).

        Args:
            audio_chunk: The raw audio data.
            overlap_mask: Boolean array where True = overlapping speech.
        """

        # 1. Mute the overlapping parts (naive approach)
        clean_audio = audio_chunk * (1 - overlap_mask)

        # 2. THE GEN-AI STEP:
        # Send 'clean_audio' to Moshi/Kyutai API.
        # Prompt: "Fill in the silenced gaps based on the context of the speaker."

        # --- PSEUDO CODE FOR API CALL ---
        # response = kyutai_api.repair(
        #     audio=clean_audio,
        #     context="This is a conversation, keep the flow of Speaker A"
        # )
        # repaired_audio = response.audio
        # return repaired_audio

        # For now, we just return the muted version (Noise Gating)
        return clean_audio

    def process_chunk(self, audio_data):
        diarization = self.get_diarization(audio_data)

        # Logic: Who is the "Main" speaker?
        # Simple heuristic: The person who spoke most in the first chunk is "Target"
        labels = diarization.labels()
        if not labels:
            return audio_data

        if self.target_speaker is None:
            self.target_speaker = labels[0]
            print(f"LOCKED onto Speaker: {self.target_speaker}")

        # Build the Mask
        # We want to keep Target, but remove frames where Target AND Others speak
        # Or remove frames where ONLY Others speak.

        n_samples = len(audio_data)
        mask = np.zeros(n_samples)

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start_sample = int(turn.start * SAMPLE_RATE)
            end_sample = int(turn.end * SAMPLE_RATE)

            # Boundary checks
            start_sample = max(0, start_sample)
            end_sample = min(n_samples, end_sample)

            if speaker != self.target_speaker:
                # Mark interference
                mask[start_sample:end_sample] = 1.0

        # Repair
        # We send the audio and the mask to the 'repair' function
        final_audio = self.repair_segment(audio_data, mask)

        return final_audio


# --- AUDIO I/O ---
q_in = queue.Queue()
q_out = queue.Queue()


def audio_callback(indata, outdata, frames, time, status):
    """Real-time callback. Puts mic data in queue, gets processed data out."""
    if status:
        print(status)
    q_in.put(indata.copy().flatten())

    # Non-blocking output (play silence if buffer empty)
    try:
        data = q_out.get_nowait()
    except queue.Empty:
        data = np.zeros(frames)

    outdata[:] = data.reshape(-1, 1)


def main():
    processor = CocktailDeMixer()

    # Rolling buffer to accumulate 2 seconds of audio
    buffer = np.zeros(CHUNK_SIZE)

    print("\n--- LISTENING (Speak into Mic) ---\n")

    # Start the stream
    with sd.Stream(
        samplerate=SAMPLE_RATE,
        blocksize=int(SAMPLE_RATE / 4),  # 250ms blocks
        channels=1,
        callback=audio_callback,
    ):

        while True:
            # 1. Get new data from mic
            try:
                new_data = q_in.get(timeout=1)
            except:
                continue

            # 2. Shift buffer and append new data
            buffer = np.roll(buffer, -len(new_data))
            buffer[-len(new_data) :] = new_data

            # 3. Process the whole buffer (Diarize + Clean)
            # Note: In a real hackathon, you optimize this to only process new parts
            cleaned_audio = processor.process_chunk(buffer)

            # 4. Output only the *new* part to the speakers
            # We take the last segment corresponding to new_data
            output_segment = cleaned_audio[-len(new_data) :]
            q_out.put(output_segment)


if __name__ == "__main__":
    main()
