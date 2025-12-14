import queue
import librosa

import numpy as np
import sounddevice as sd
import torch
from pyannote.audio import Pipeline
import collections

from pyannoteai.sdk import Client

import os

key_diarization = os.getenv('KEYPYANNOTE', default=None)

SAMPLE_RATE = 48000
CHUNK_DURATION = 2.0 # How to choose this
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
overlap_duration = 0.5 # min duration of overlap to run the separation

class DeMixer:
    def __init__(self):
        print("Loading Pyannote Pipeline (this takes a moment)...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # call api
        self.client_diarization = Client(key_diarization)

        print(f"Pipeline ready on {self.device}")
        self.target_speaker = None


        # etape 1: enregistrer la voix


    def get_diarization(self, audio_buffer):
        """
        Runs Pyannote on the buffer to find who is speaking when.
        """
        # Pyannote expects a Tensor (channels, time)
        audio_tensor = torch.tensor(audio_buffer).float().unsqueeze(0).to(self.device)

        # submit a diarization job
        job_id = self.client_diarization.diarize(self.client_diarization.upload(audio_buffer))

        # retrieve diarization
        diarization = self.client_diarization.retrieve(job_id)

        # this variable is a list of dict
        return diarization['output']['diarization']

    def repair_segment(self, audio_chunk, overlap_mask):
        """
        Only separats the parts where ovelap mask is 1.

        Args:
            audio_chunk: The raw audio data.
            overlap_mask: Boolean array where True = overlapping speech.

        """

        ######################################
        ########### SEPARATION ###############
        ######################################
        # After separation how to find the skpeaker we want to keep

        # Mute the overlapping parts (naive approach)
        clean_audio = audio_chunk * (1 - overlap_mask)

        return clean_audio

    def process_chunk(self, audio_data):
        audio_data = np.ascontiguousarray(audio_data)
        audio_16k = librosa.resample(audio_data, orig_sr=48000, target_sr=16000)

        segments = self.get_diarization(audio_16k)

        ####################################################
        ####### SOME LOGIC TO DEFINE THE MAIN SPEAKER ######
        ####################################################
        if self.target_speaker is None:
            self.target_speaker = None
            print(f"LOCKED onto Speaker: {self.target_speaker}")

        # Build the Mask
        # We want to keep Target, but process the frames where Target AND Others speak (seperation etc)
        # Or remove frames where ONLY Others speak.

        n_samples = len(audio_data)
        overlap_mask = np.zeros(n_samples, dtype=np.float32)

        #####################################
        ### BUILD THE MASK ? ################
        #####################################

        for i, seg_i in enumerate(segments):
            # only overlaps involving target speaker
            if seg_i["speaker"] != self.target_speaker:
                continue

            for j, seg_j in enumerate(segments):
                if i != j:
                    # intersection
                    overlap_start = max(seg_i["start"], seg_j["start"])
                    overlap_end = min(seg_i["end"], seg_j["end"])

                    if overlap_start >= overlap_end:
                        continue  # no overlap
                    else:
                        start_sample = int(overlap_start * SAMPLE_RATE)
                        end_sample = int(overlap_end * SAMPLE_RATE)

                        start_sample = max(0, start_sample)
                        end_sample = min(n_samples, end_sample)

                        overlap_mask[start_sample:end_sample] = 1.0

        assert len(overlap_mask) == len(audio_data)
        final_audio = self.repair_segment(audio_data, overlap_mask)

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
    processor = DeMixer()

    devices = sd.query_devices()
    default_input = sd.default.device[0]

    print("\n--- Available Devices ---")
    print(devices)
    print(f"\nUsing Input Device Index: {default_input}")

    device = sd.default.device[0]  # input device
    info = sd.query_devices(device)
    print(info)

    INPUT_DEV = 7
    OUTPUT_DEV = 7

    audio_buffer = collections.deque(maxlen=CHUNK_SIZE)

    audio_buffer.extend(np.zeros(CHUNK_SIZE, dtype=np.float32))

    print(f"\nUsing Devices -> In: {INPUT_DEV}, Out: {OUTPUT_DEV}")
    print("\n--- LISTENING ---")
    with sd.Stream(
        device=(INPUT_DEV, OUTPUT_DEV),
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SIZE,
        channels=1,
        callback=audio_callback,
    ):

        while True:
            try:
                new_data = q_in.get(timeout=1)
                print(new_data)
            except queue.Empty:
                print("Warning: No audio data received from mic...")
                continue

            audio_buffer.extend(new_data)
            buff = np.array(audio_buffer, dtype=np.float32)

            ##############################################
            ######### SOME LOGIC FOR PREPROCESSING ##########
            ##############################################


            cleaned_audio = processor.process_chunk(buff)


            ###############################################
            ########## GIVE TO MOSHI OR ELSE ? ############
            ###############################################
            q_out.put(cleaned_audio[-CHUNK_SIZE:])


if __name__ == "__main__":
    main()
