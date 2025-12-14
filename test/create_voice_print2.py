import time
import sounddevice as sd
import scipy.io.wavfile as wav
import requests
import os
from pyannoteai.sdk import Client

# --- 1. CONFIGURATION ---
API_KEY = 'sk_7e7cf9e9186c464bb8fe489b0c21af44'
client = Client(api_key=API_KEY)

# Files
VOICEPRINT_FILE = "my_voiceprint.txt"
# FILENAME = "mixture_audio.wav"
FILENAME="/home/saimeur/workspace/projet_perso/voice_ai/mixture_3_people.wav"

# Settings
SAMPLE_RATE = 48000
DURATION = 15.0  # Recording time for the meeting
TARGET_NAME = "My Target Speaker" # The name you want to see in the results

def identify_speaker():
    # --- 2. LOAD VOICEPRINT ---
    try:
        with open(VOICEPRINT_FILE, "r") as f:
            my_voiceprint_code = f.read().strip()
        print(f"✅ Loaded voiceprint from {VOICEPRINT_FILE}")
    except FileNotFoundError:
        print(f"❌ Error: Could not find {VOICEPRINT_FILE}. Run the first script first!")
        return

    # --- 3. RECORD MEETING ---
    # print(f"🎤 Recording conversation for {DURATION} seconds...")
    # audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    # sd.wait()
    # wav.write(FILENAME, SAMPLE_RATE, audio_data)
    # print("✅ Recording saved. Uploading...")

    # --- 4. UPLOAD AUDIO ---
    try:
        media_url = client.upload(FILENAME)
        print(f"Uploaded to: {media_url}")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return

    # --- 5. START IDENTIFICATION JOB ---
    print("Requesting identification...")
    
    payload = {
        "url": media_url,
        "voiceprints": [
            {
                "label": TARGET_NAME, 
                "voiceprint": my_voiceprint_code
            }
        ],
        # Optional: Set confidence threshold (0.0 to 1.0)
        # "matching": { "threshold": 0.6 } 
    }

    response = requests.post(
        "https://api.pyannote.ai/v1/identify",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json=payload
    )

    if response.status_code != 200:
        print("\n❌ API ERROR!")
        print(f"Status Code: {response.status_code}")
        print(f"Server Message: {response.text}")
        return

    job_id = response.json()['jobId']
    print(f"Job started: {job_id}")

    # --- 6. WAIT FOR RESULT ---
    while True:
        # Check the JOBS endpoint (Same fix as before)
        job_response = requests.get(
            f"https://api.pyannote.ai/v1/jobs/{job_id}",
            headers={"Authorization": f"Bearer {API_KEY}"}
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
        time.sleep(2)

if __name__ == "__main__":
    identify_speaker()