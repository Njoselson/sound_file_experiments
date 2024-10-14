# Sound File Experiments

This repository contains a collection of Python scripts for experimenting with sound file generation, manipulation, and transcription. The project utilizes various libraries and tools to create, modify, and analyze audio files.

## Features

- Generate high-pitched sound files
- Combine generated sounds with spoken words
- Transcribe audio files using OpenAI's Whisper model
- Experiment with different sound parameters (duration, frequency, volume, and waveform shape)

## Dependencies

- NumPy
- SciPy
- PyDub
- OpenAI Whisper
- FFmpeg

## Usage

1. Install the required dependencies:
   ```
   pip install numpy scipy pydub openai-whisper
   ```

2. Ensure FFmpeg is installed on your system. On macOS, you can use Homebrew:
   ```
   brew install ffmpeg
   ```

3. Run the various Python scripts to generate and manipulate sound files.

## Shortening Audio Files

To shorten an audio file, you can use the following FFmpeg command:
```
ffmpeg -i input_file.wav -t 10 output_file.wav
```
This command will shorten the audio file to 10 seconds.

## Generating High-Pitched Sounds

To generate a high-pitched sound, you can use the following Python script:
```
python sound_file_generation.py
```
This script will generate a high-pitched sound with a duration of 1 second, a frequency of 1000 Hz, a volume of 1.0, and a sine waveform shape.

## Transcribing Audio Files

To transcribe an audio file, you can use the following Python script:
```
python sound_file_transcription.py
```
This script will transcribe the audio file and print the transcription to the console.

## Experimenting with Different Sound Parameters

You can experiment with different sound parameters by modifying the parameters in the Python scripts. For example, you can change the duration, frequency, volume, and waveform shape of the sound by modifying the parameters in the `generate_high_pitched_sound` function.   
