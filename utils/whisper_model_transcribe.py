from openai import OpenAI

# load dotenv
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI()

def transcribe_audio(audio_file_path):
    with open(audio_file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1"
        )
    return response


audio_file_path = "results/best_sound_r0.0000.wav"
transcription = transcribe_audio(audio_file_path)
print(transcription)
