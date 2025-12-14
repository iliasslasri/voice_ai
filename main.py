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
TARGET_NAME = "target_name"  # The speaker label you want to assign
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

    def target_speaker_detection(self, audio_chunk):
        # upload conversation file
        temp_chunck = "temp_chunk.wav"
        wav_file = wave.open(temp_chunck, 'w')
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit PCM
        wav_file.setframerate(SAMPLE_RATE)
        # Convert float32 -> int16
        audio_int16 = np.int16(audio_chunk * 32767)
        wav_file.writeframes(audio_int16.tobytes())
        wav_file.close()
        media_url = self.client.upload(temp_chunck)
        data = {
            "url": media_url,
            "voiceprints": [
                {
                    "label": TARGET_NAME, # The speaker label you want to assign
                    "voiceprint": self.target_speaker  # Replace with actual voiceprint
                },
                # Add more voiceprints as needed
            ],
            # Optional matching parameters
            "matching": {
                "threshold": 50,  # Only match if confidence is 50% or higher
                "exclusive": True  # Prevent multiple speakers matching same voiceprint
            }
        }

        import requests
        url = "https://api.pyannote.ai/v1/identify"

        headers = {"Authorization": f"Bearer {access_key}", "Content-Type": "application/json"}

        response = requests.post(url, headers=headers, json=data)

        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
        else:
            print(response.json())


        if response.status_code != 200:
            print("\n❌ API ERROR!")
            print(f"Status Code: {response.status_code}")
            print(f"Server Message: {response.text}")


        job_id = response.json()['jobId']
        print(f"Job started: {job_id}")

        # --- 6. WAIT FOR RESULT ---
        while True:
            # Check the JOBS endpoint (Same fix as before)
            job_response = requests.get(
                f"https://api.pyannote.ai/v1/jobs/{job_id}",
                headers={"Authorization": f"Bearer {access_key}"}
            )
            
            job_data = job_response.json()
            
            if 'status' not in job_data:
                print("❌ Unexpected response:", job_data)
                break
                
            status = job_data['status']

            if status == "succeeded":
                print("\n🎉 ANALYSIS COMPLETE!")
                print("-" * 50)
                
                # The results are in output -> identification
                segments = job_data['output']['identification']
                
                found_target = False
                for segment in segments:
                    speaker = segment['speaker']
                    start = segment['start']
                    end = segment['end']
                    
                    # If the AI recognized the voiceprint, 'speaker' will be TARGET_NAME
                    # If not, it will be generic (SPEAKER_00, SPEAKER_01...)
                    if speaker == TARGET_NAME:
                        print(f"🟢 {speaker} found: {start:.1f}s -> {end:.1f}s")
                        found_target = True
                    else:
                        print(f"⚪ Unknown ({speaker}): {start:.1f}s -> {end:.1f}s")
                
                if not found_target:
                    print(f"⚠️ {TARGET_NAME} was not detected in this audio.")
                    
                print("-" * 50)
                break
                
            elif status == "failed":
                print("\n❌ Job failed.")
                print(job_data)
                break
            
            print("Processing...", end="\r")


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
    voiceprint = DeMixer.client.retrieve(job_id)
    # speaker_embed variable
    DeMixer.target_speaker = voiceprint['output']['voiceprint']

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
