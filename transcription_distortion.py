import os
import warnings
import string
import math
from dotenv import load_dotenv
from collections import Counter

import numpy as np
from scipy.io import wavfile
from scipy.optimize import differential_evolution
from pydub import AudioSegment
import whisper
from difflib import SequenceMatcher
from skopt import gp_minimize
import sounddevice as sd
import torch
from openai import OpenAI
import whisper


# Suppress the warning about FP16 not being supported on CPU
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

load_dotenv()

client = OpenAI()
LOCAL_WHISPER_MODEL = whisper.load_model("base")

# Set the path to the FFmpeg executables (if needed)
AudioSegment.converter = "/opt/homebrew/bin/ffmpeg"
AudioSegment.ffmpeg = "/opt/homebrew/bin/ffmpeg"
AudioSegment.ffprobe = "/opt/homebrew/bin/ffprobe"

# Load the Whisper model
model = whisper.load_model("base")

audio_folder = "audio"

# Initialize global variables
global best_reward, best_params, transcription_type
best_reward = float('-inf')
best_params = None
transcription_type = "open_ai"

# Load the spoken word audio
spoken_word = AudioSegment.from_wav(f"{audio_folder}/word.wav")
two_seconds_silence = AudioSegment.silent(duration=2000)
word_with_silence = spoken_word + two_seconds_silence

# Parameters to iterate over
durations = [ 0.0001, 0.001, 0.01, 0.1,] 
frequencies = [20000, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000]
volumes = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
shapes = ['sine', 'square', 'sawtooth']

def transcribe_audio_openai_whisper(audio_file_path):
    with open(audio_file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1"
        )
    return response

def transcribe_open_source_whisper(audio_file_path):
    result = LOCAL_WHISPER_MODEL.transcribe(audio_file_path)
    return result
    

# Function to generate high-pitched sound
def generate_high_pitched_sound(duration, freq, volume, shape):
    t = np.linspace(0, duration, int(44100 * duration), False)
    if shape == 'sine':
        note = np.sin(freq * t * 2 * np.pi)
    elif shape == 'square':
        note = np.sign(np.sin(freq * t * 2 * np.pi))
    elif shape == 'sawtooth':
        note = 2 * (freq * t - np.floor(0.5 + freq * t))
    else:
        raise ValueError("Invalid shape. Choose 'sine', 'square', or 'sawtooth'.")
    
    note = note * volume
    audio = note * (2**15 - 1) / np.max(np.abs(note))
    audio = audio.astype(np.int32)
    return audio

# Function to generate sound based on params
def generate_sound(duration, freq, volume, shape):
    high_pitched_audio = generate_high_pitched_sound(duration, freq, volume, shape)
    wavfile.write("temp_sound.wav", 44100, high_pitched_audio)
    high_pitched = AudioSegment.from_wav("temp_sound.wav")
    combined = high_pitched + word_with_silence
    combined.export("temp_sound.wav", format="wav")
    return combined

def calculate_weirdness(transcription):
    # Convert to lowercase for consistency
    text = transcription.lower()
    
    # Calculate factors contributing to weirdness
    length_factor = min(len(text) / 2000000000, 1)  # Reward longer transcriptions, up to a point
    non_ascii_factor = len([c for c in text if ord(c) > 127]) / len(text) if len(text) > 0 else 0
    unique_chars_factor = len(set(text)) / len(text) if len(text) > 0 else 0
    
    # Count the frequency of each character
    char_frequencies = Counter(text)
    entropy = sum([-freq/len(text) * math.log2(freq/len(text)) for freq in char_frequencies.values()]) if len(text) > 0 else 0
    entropy_factor = min(entropy / 4, 1)  # Normalize entropy, assuming 4 bits as a high entropy
    
    # Combine factors
    weirdness = (length_factor + non_ascii_factor + unique_chars_factor + entropy_factor) / 4
    return weirdness

def calculate_reward(new_transcription, previous_transcriptions):
    if not new_transcription:
        return 0  # No reward for empty transcriptions

    if not previous_transcriptions:
        return 1.0  # Maximum reward for the first non-empty transcription
    
    # Calculate similarity to each previous transcription
    similarities = [SequenceMatcher(None, new_transcription, prev).ratio() 
                    for prev in previous_transcriptions if prev]
    
    if not similarities:
        return 1.0  # Maximum reward if all previous transcriptions were empty
    
    # The reward is inverse to the maximum similarity
    max_similarity = max(similarities)
    reward = 1.0 - max_similarity
    
    # Bonus for longer transcriptions
    length_bonus = min(len(new_transcription) / 20, 1)  # Cap at 1 for transcriptions of 20+ characters
    
    return reward + length_bonus


def objective(params):
    global best_reward, best_params, transcription_type
    duration, freq, volume, shape_param = params
    shape = ['sine', 'square', 'sawtooth'][int(shape_param)]
    
    # Generate sound
    sound = generate_sound(duration, freq, volume, shape)
    
    if transcription_type == "open_source":
        transcription = transcribe_open_source_whisper("temp_sound.wav")
        transcription = transcription["text"].strip().lower()
    elif transcription_type == "open_ai":
        transcription = transcribe_audio_openai_whisper("temp_sound.wav")
        transcription = transcription.text.strip().lower()
    else:
        raise ValueError("Invalid transcription type specified")

    
    # Calculate reward
    if transcription.lower() == "this is":
        reward = 0.0
    else:
        reward = calculate_weirdness(transcription)
    
    print(f"Trying sound. Parameters: duration={duration:.6f}, freq={freq:.2f}, volume={volume:.2f}, shape={shape}")
    print(f"Transcription: {transcription}")
    print(f"Reward: {reward:.4f}")
    
    if reward > 0 and ('best_reward' not in globals() or reward > best_reward):
        best_reward = reward
        best_params = params
        print(f"New best reward: {best_reward:.4f}")
        # save best sound parameters:
        # save best results to directory results
        os.makedirs("results", exist_ok=True)
        with open(f"results/best_parameters_{reward:.4f}.txt", "w") as f:
            f.write(f"duration={best_params[0]:.6f}, freq={best_params[1]:.2f}, volume={best_params[2]:.2f}, shape={['sine', 'square', 'sawtooth'][int(best_params[3])]}")

        
        # Save the best sound
        best_sound_filename = f"results/best_sound_r{reward:.4f}.wav"
        sound.export(best_sound_filename, format="wav")
        print(f"Saved best sound to: {best_sound_filename}")
    
    return -reward  # Note the negative sign for minimization

# Define the search space
space = [(0.01, 0.5),          # duration
         (20, 20000),          # freq
         (0.1, 150.0),           # volume
         (0, 2.99)]            # shape (will be converted to int)



# Run the optimization
result = gp_minimize(objective, space, n_calls=500, verbose=True)

# Print results
print("\nOptimization finished.")
print(f"Best parameters: duration={best_params[0]:.6f}, freq={best_params[1]:.2f}, volume={best_params[2]:.2f}, shape={['sine', 'square', 'sawtooth'][int(best_params[3])]}")
print(f"Best reward: {best_reward}")

# Generate and play the best sound
best_sound = generate_sound(best_params[0], best_params[1], best_params[2], ['sine', 'square', 'sawtooth'][int(best_params[3])])
sd.play(best_sound, samplerate=44100)
sd.wait()

# Transcribe the best sound
audio_data = torch.from_numpy(best_sound).float()
final_result = model.transcribe(audio_data, fp16=False)
# TODO: Refactor this to use the same transcription function as the optimization loop
if transcription_type == "open_source":
    final_transcription = final_result["text"].strip().lower()
elif transcription_type == "open_ai":
    final_transcription = final_result.text.strip().lower()
print(f"Final transcription: {final_transcription}")
