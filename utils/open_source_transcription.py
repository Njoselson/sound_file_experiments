import whisper
from pydub import AudioSegment
from pathlib import Path

# Set the path to the FFmpeg executables using the parent directory
ffmpeg_path = Path("/opt/homebrew/bin")
AudioSegment.converter = str(ffmpeg_path / "ffmpeg")
AudioSegment.ffmpeg = str(ffmpeg_path / "ffmpeg")
AudioSegment.ffprobe = str(ffmpeg_path / "ffprobe")

# ... your existing code for generating high-pitched sound ...

# Load the Whisper model
model = whisper.load_model("base")

# Set the correct path to the audio folder
current_file = Path(__file__)
project_root = current_file.parent.parent
audio_folder = project_root / "audio"

# Transcribe the spoken word audio file
result = model.transcribe(str(audio_folder / "word.wav"))

print("Transcription:", result["text"])
