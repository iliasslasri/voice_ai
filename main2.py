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
        clean_audio = None
        
        # # Mute the overlapping parts (naive approach)
        # clean_audio = audio_chunk * (1 - overlap_mask)
        
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

        # We need to match the target to the separate source audio

        return final_audio


# --- AUDIO I/O ---
q_in = queue.Queue()
q_out = queue.Queue()

# ... keep your imports and DeMixer class as they are ...

# --- UPDATED AUDIO SETTINGS ---
# We stream in small bits (low latency) but process in big chunks (better AI accuracy)
STREAM_BLOCK_SIZE = 2048  
curr_buffer = [] # Temporary list to build up our 2-second chunk

def audio_callback(indata, outdata, frames, time, status):
    """
    Fast callback! Puts small chunks into the queue.
    """
    if status:
        print(f"Stream Status: {status}")
    
    # Put the small chunk (2048 samples) into the queue
    q_in.put(indata.copy().flatten())

    # Try to get processed audio to play
    try:
        # We grab exactly 'frames' amount of audio to play back
        data = q_out.get_nowait()
    except queue.Empty:
        # If no processed audio is ready, play silence (don't crash)
        data = np.zeros(frames)

    outdata[:] = data.reshape(-1, 1)

def main():
    processor = DeMixer()

    # --- DEVICE SELECTION (Auto-select to avoid ID 17 crash) ---
    try:
        default_in = sd.query_devices(kind='input')
        default_out = sd.query_devices(kind='output')
        print(f"Using Input: {default_in['name']}")
    except:
        print("Using system defaults.")

    # Main buffer for the 2-second sliding window
    audio_buffer = collections.deque(maxlen=CHUNK_SIZE)
    # Fill it with silence first to avoid "Empty Slice" errors at startup
    audio_buffer.extend(np.zeros(CHUNK_SIZE, dtype=np.float32))

    print("\n--- LISTENING (Press Ctrl+C to stop) ---")
    
    # We use a small blocksize for the stream so it feels responsive
    with sd.Stream(samplerate=SAMPLE_RATE, blocksize=STREAM_BLOCK_SIZE,
                   channels=1, callback=audio_callback):
        
        # Local buffer to accumulate the small stream blocks until we have 2 seconds
        accumulated_samples = [] 

        while True:
            try:
                # 1. Get small chunk from Mic
                new_data = q_in.get(timeout=2) 
                
                # 2. Add to our accumulator
                accumulated_samples.extend(new_data)

                # 3. DO WE HAVE ENOUGH DATA TO PROCESS? (2 Seconds worth)
                if len(accumulated_samples) >= CHUNK_SIZE:
                    
                    # Convert to numpy array
                    chunk_to_process = np.array(accumulated_samples[:CHUNK_SIZE])
                    
                    # Reset accumulator (overlapping window? For now, we do non-overlapping for speed)
                    accumulated_samples = accumulated_samples[CHUNK_SIZE:] 

                    # Update the main rolling buffer
                    audio_buffer.extend(chunk_to_process)
                    buff = np.array(audio_buffer, dtype=np.float32)

                    # --- LATENCY HACK: DROP OLD FRAMES ---
                    # If the queue is huge, we are falling behind.
                    # Skip processing this chunk to catch up.
                    if q_in.qsize() > 10: 
                        print("!! SYSTEM OVERLOAD - SKIPPING FRAME TO CATCH UP !!")
                        q_out.put(np.zeros(CHUNK_SIZE)) # Play silence while catching up
                        continue

                    # --- SILENCE CHECK (Optimization) ---
                    # If volume is super low, don't waste GPU power processing it
                    vol = np.linalg.norm(buff) / len(buff)
                    if vol < 0.0001: 
                        # Just pass the silence through
                        q_out.put(buff)
                        continue

                    # --- HEAVY PROCESSING ---
                    try:
                        cleaned_audio = processor.process_chunk(buff)
                        
                        # Handle case where process_chunk returns None or error
                        if cleaned_audio is None:
                            cleaned_audio = buff

                        # Send to speakers
                        # We need to slice it to match the stream block logic, 
                        # but for simplicity, we push the whole chunk.
                        # (Note: In a perfect system, you'd chop this back into 2048 blocks)
                        
                        # For this specific loop structure, we push the whole 2s
                        # But the callback consumes it in small bites. 
                        # We need to feed the callback queue chunk-by-chunk.
                        
                        # Slicing the 2s processed audio back into small blocks for the callback
                        for i in range(0, len(cleaned_audio), STREAM_BLOCK_SIZE):
                            small_block = cleaned_audio[i : i + STREAM_BLOCK_SIZE]
                            # Pad if necessary
                            if len(small_block) < STREAM_BLOCK_SIZE:
                                small_block = np.pad(small_block, (0, STREAM_BLOCK_SIZE - len(small_block)))
                            q_out.put(small_block)

                    except Exception as e:
                        print(f"Processing Error: {e}")
                        q_out.put(np.zeros(STREAM_BLOCK_SIZE))

            except queue.Empty:
                continue
            except KeyboardInterrupt:
                print("Stopping...")
                break

if __name__ == "__main__":
    main()