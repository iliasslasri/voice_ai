import queue
import librosa

import numpy as np
import sounddevice as sd
import torch
from pyannote.audio import Pipeline
import collections

SAMPLE_RATE = 48000
CHUNK_DURATION = 2.0 # How to choose this
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
overlap_duration = 0.5 # min duration of overlap to run the separation

class DeMixer:
    def __init__(self):
        print("Loading Pyannote Pipeline (this takes a moment)...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.diarization = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1"
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

        diarization = self.diarization(
            {"waveform": audio_tensor, "sample_rate": SAMPLE_RATE}
        )
        return diarization

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
        diarization = self.get_diarization(audio_16k)
        
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
        overlap_mask = np.zeros(n_samples)

        #####################################
        ### BUILD THE MASK ? ################
        #####################################

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
