import os
import time
import sounddevice as sd
import scipy.io.wavfile as wav
from pyannoteai.sdk import Client

# 1. SETUP: Ensure you have your API Key
# You can hardcode it for testing: Client(api_key="your_key_here")
# Or use the environment variable as you tried before:
# api_key = os.getenv("PYANNOTE_API_KEY") 
api_key = 'sk_7e7cf9e9186c464bb8fe489b0c21af44'
client = Client(api_key=api_key) if api_key else Client()
print(f'API KEY : {api_key}')


SAMPLE_RATE = 48000
DURATION = 5.0 
FILENAME = "recording.wav"

def record_and_diarize():
    # --- STEP 1: RECORD ---
    print(f"Recording for {DURATION} seconds...")
    audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    wav.write(FILENAME, SAMPLE_RATE, audio_data)
    print("Recording saved. Uploading...")

    try:
        # --- STEP 2: UPLOAD & START JOB ---
        # The SDK handles the upload to Pyannote's servers
        media_url = client.upload(FILENAME)
        job_id = client.diarize(media_url)
        
        print(f"Job started (ID: {job_id})")
        print("Waiting for results...")

        # --- STEP 3: POLLING LOOP (Required) ---
        # You must loop and ask "Are you done?" until status is 'succeeded'
        while True:
            result = client.retrieve(job_id) # Returns a Dictionary
            status = result['status']        # Fix: Use brackets, not .status
            
            if status == "succeeded":
                print("Processing complete! ✅")
                break
            elif status == "failed":
                print("Job failed. ❌")
                return
            
            # Wait 2 seconds before checking again to avoid spamming
            time.sleep(2)

        # --- STEP 4: PROCESS RESPONSE ---
        # Fix: Use brackets ['output'] instead of .output
        diarization = result['output']['diarization']
        
        for segment in diarization:
            # Fix: Use brackets for segment data too
            speaker = segment['speaker']
            start = segment['start']
            end = segment['end']
            print(f"Speaker {speaker}: {start:.1f}s - {end:.1f}s")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    record_and_diarize()