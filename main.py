import queue
import librosa
import time
import numpy as np
import sounddevice as sd
import torch
from pyannote.audio import Pipeline
import collections
import wave
from pyannoteai.sdk import Client

SAMPLE_RATE = 48000
CHUNK_DURATION = 2.0 # How to choose this
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
THRESHOLD = 0.01    # Voice detection threshold
RECORD_SECONDS = 5  # 5 second recording for voiceprint acquisition
access_key = 'sk_7e7cf9e9186c464bb8fe489b0c21af44'
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
        self.client = Client(access_key)

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
        
        # ####################################################
        # ####### SOME LOGIC TO DEFINE THE MAIN SPEAKER ######
        # ####################################################
        # if self.target_speaker is None:
        #     self.target_speaker = None
        #     print(f"LOCKED onto Speaker: {self.target_speaker}")
        # else:
        #     self.detect_speaker(diarization)
        
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

    
    # Get speaker voiceprint 
    callback.start_time = None
    TEMP_FILE = "main_speaker.wav"

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback, dtype="float32"):
        sd.sleep(60_000)  # maximum wait 60 seconds for a speaker
    audio_np = np.concatenate(recorded_audio, axis=0)
    # Save temporary WAV
    wav_file = wave.open(TEMP_FILE, 'w')
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)  # 16-bit PCM
    wav_file.setframerate(SAMPLE_RATE)
    # Convert float32 -> int16
    audio_int16 = np.int16(audio_np * 32767)
    wav_file.writeframes(audio_int16.tobytes())
    wav_file.close()

    # Upload audio
    media_url = DeMixer.client.upload(TEMP_FILE)
    # Submit embedding job
    job_id = DeMixer.client.voiceprint(media_url)
    # Retrieve embedding
    embedding_result = DeMixer.client.retrieve(job_id)
    # speaker_embed variable
    DeMixer.target_speaker = embedding_result['output']['voiceprint']

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


def detect_voice(indata):
    """Return True if voice is detected in the frame"""
    volume_norm = np.linalg.norm(indata) / len(indata)
    return volume_norm > THRESHOLD

def callback(indata, frames, time_info, status):
    global recorded_audio, recording_started
    if detect_voice(indata) and not recording_started:
        print("Speaker detected! Starting 10-second recording...")
        recording_started = True
        recorded_audio.append(indata.copy())
        callback.start_time = time.time()
    elif recording_started:
        recorded_audio.append(indata.copy())
        # Stop after 10 seconds
        if time.time() - callback.start_time >= RECORD_SECONDS:
            raise sd.CallbackStop()

if __name__ == "__main__":
    main()
