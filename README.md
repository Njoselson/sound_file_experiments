# Sound File Experiments

This repository contains a collection of Python scripts for experimenting with adversarial audio generation. The project aims to create sound files that, when transcribed by speech-to-text models, produce outputs significantly different from the actual spoken words. It utilizes various libraries and tools to generate, manipulate, and analyze these adversarial audio files. Additionally, the project employs Bayesian optimization techniques to fine-tune the parameters of the sound distribution, with the goal of producing increasingly unusual and unexpected transcriptions from the speech-to-text model.

- Transcribe audio files using OpenAI's Whisper model or open-source Whisper model
- Experiment with different sound parameters (duration, frequency, volume, and waveform shape)
- Utils for manipulating audio files

## Dependencies

## Usage

1. Install the required dependencies:
   ```
   python3 -m venv .sound_file_experiments
   source .sound_file_experiments/bin/activate
   pip install numpy scipy pydub openai-whisper scikit-optimize python-dotenv sounddevice
   ```

2. Ensure FFmpeg is installed on your system. On macOS, you can use Homebrew:
   ```
   brew install ffmpeg
   ```
3. Add your OpenAI API key to the `.env` file

4. Run the training of the transcription distortion sound
```
python transcription_distortion.py
```


# Sound File Utils

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
