import sounddevice as sd
import numpy as np

# --- SETTINGS ---
SAMPLE_RATE = 48000
BLOCK_SIZE = 1024
DTYPE = 'float32' # standard for Python audio

# Based on your logs:
INPUT_DEV = 7   # sysdefault
OUTPUT_DEV = 7  # sysdefault (Try 8 or 1 if this fails)

def callback(indata, outdata, frames, time, status):
    if status:
        print(f"Status: {status}")
    
    # Check volume level
    volume = np.linalg.norm(indata) * 10
    
    # Print the raw volume number to debug
    print(f"Input Level: {volume:.4f}")
    
    # Pass audio directly to speakers
    outdata[:] = indata

print(f"Testing Audio Passthrough (In:{INPUT_DEV}, Out:{OUTPUT_DEV})...")
print("Speak into mic. You should see 'Input Level' go up.")

try:
    with sd.Stream(
        device=(INPUT_DEV, OUTPUT_DEV),
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        dtype=DTYPE,
        channels=1,
        callback=callback
    ):
        input("Press Enter to Stop...")
except Exception as e:
    print(f"Error: {e}")