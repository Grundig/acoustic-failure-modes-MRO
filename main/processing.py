from scipy.signal import spectrogram

from reading_data import load_audio

def load():
    audio, sample_rate = load_audio("recording.aac")

    return audio, sample_rate


def plot_spectrogram(audio, fs):
    frequencies, times, spectrum = spectrogram(audio, fs=fs)

    return frequencies, times, spectrum


def main():
    audio, sample_rate = load_audio()
    frequencies, times, spectrum = spectrogram(audio, fs=sample_rate)

    return frequencies, times, spectrum


if __name__ == "__main__":
    main()
