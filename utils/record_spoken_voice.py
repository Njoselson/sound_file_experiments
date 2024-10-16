import sounddevice as sd
import numpy as np
from scipy.io import wavfile

def record_audio(duration, sample_rate=44100):
    print("Recording...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    print("Recording complete.")
    return recording

def save_wav(filename, audio, sample_rate):
    wavfile.write(filename, sample_rate, audio)
    print(f"Audio saved as {filename}")

if __name__ == "__main__":
    duration = 5  # Recording duration in seconds
    sample_rate = 44100  # Sample rate in Hz
    filename = "spoken_word.wav"

    audio = record_audio(duration, sample_rate)
    save_wav(filename, audio, sample_rate)
