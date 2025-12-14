import time
import sounddevice as sd
import scipy.io.wavfile as wav
import requests
import os
from pyannoteai.sdk import Client

# --- 1. SETUP ---
API_KEY = 'sk_7e7cf9e9186c464bb8fe489b0c21af44' 
client = Client(api_key=API_KEY)

SAMPLE_RATE = 48000
DURATION = 20.0
# FILENAME = "reference.wav"
FILENAME="reference.wav"
# Name of the text file where we will save the code
VOICEPRINT_FILE = "my_voiceprint.txt" 

def create_voiceprint():
    # --- 2. RECORD ---
    # print(f"🎤 Speak alone for {DURATION} seconds...")
    # audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    # sd.wait()
    # wav.write(FILENAME, SAMPLE_RATE, audio_data)
    # print("✅ Recording saved. Uploading...")

    # --- 3. UPLOAD ---
    try:
        media_url = client.upload(FILENAME)
        print(f"Uploaded to: {media_url}")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return

    # --- 4. CREATE VOICEPRINT ---
    print("Requesting voiceprint creation...")
    
    response = requests.post(
        "https://api.pyannote.ai/v1/voiceprint",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={"url": media_url}
    )

    if response.status_code != 200:
        print("\n❌ API ERROR!")
        print(f"Status Code: {response.status_code}")
        print(f"Server Message: {response.text}")
        return

    data = response.json()
    job_id = data['jobId']
    print(f"Job started: {job_id}")

    # --- 5. WAIT FOR RESULT ---
    while True:
        # Check the JOBS endpoint
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
            print("\n🎉 SUCCESS!")
            
            # Extract the long code
            code = job_data['output']['voiceprint']
            
            # --- NEW: SAVE TO TXT FILE ---
            with open(VOICEPRINT_FILE, "w") as f:
                f.write(code)
            
            print(f"✅ Voiceprint saved to: {VOICEPRINT_FILE}")
            print(f"Code preview: {code[:20]}...") 
            break
            
        elif status == "failed":
            print("\n❌ Job failed.")
            print(job_data)
            break
        
        print("Processing...", end="\r")
        time.sleep(2)

if __name__ == "__main__":
    create_voiceprint()