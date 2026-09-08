from scipy.signal import spectrogram
from tkinter import filedialog
from afm.reading_data import load_audio
from matplotlib import pyplot as plt
from afm.helper_functions import normalize

def load():
    audio, sample_rate = load_audio(filedialog.askopenfilename())

    return audio, sample_rate


def plot_spectrogram(audio, fs, vmin=None, vmax=None):
    frequencies, times, spectrum = spectrogram(audio, fs=fs)
    spectrum = normalize(spectrum)
    plt.imshow(spectrum, aspect='auto', origin='lower', extent=[times.min(), times.max(), frequencies.min(), frequencies.max()], vmin=vmin, vmax=vmax)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram')
    plt.colorbar()
    plt.show()


def main():
    audio, sample_rate = load()
    plot_spectrogram(audio, sample_rate, vmax=0.01)


if __name__ == "__main__":
    main()
