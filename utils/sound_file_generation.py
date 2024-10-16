import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment

def generate_sound_file():
    # Set the parameters
    duration = 1  # Duration of the sound in seconds
    sample_rate = 44100  # Standard sample rate (CD quality)
    frequency = 15000  # Frequency in Hz (very high-pitched)

    # Generate the time array
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # Generate the sine wave
    audio = np.sin(8 * np.pi * frequency * t)

    # Normalize the audio to 16-bit range
    audio = (audio * 32767).astype(np.int16)

    # Save as WAV file
    wavfile.write("high_pitched_sound.wav", sample_rate, audio)

    print("High-pitched sound file created: high_pitched_sound.wav")

    print("Now appending a spoken word to the high-pitched sound file")

    # Load the high-pitched sound file
    high_pitched_sound = AudioSegment.from_wav("high_pitched_sound.wav")

    # Load the spoken word file
    spoken_word = AudioSegment.from_wav("spoken_word.wav")

    # Append the spoken word to the high-pitched sound
    combined_sound = high_pitched_sound + spoken_word

    # Export the combined sound to a new WAV file
    combined_sound.export("sound_file_experiment.wav", format="wav")

    print("Combined sound file created with words and noise")

def main():
    generate_sound_file()

if __name__ == "__main__":
    main()
