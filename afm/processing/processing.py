from tkinter import filedialog
from afm.importers.reading_data import load_audio
from afm.visualisation.plots import Plots


def load():
    audio, sample_rate = load_audio(filedialog.askopenfilename())

    return audio, sample_rate


def main():
    audio, sample_rate = load()
    Plots.plot_spectrogram(audio, sample_rate, vmax=0.01)


if __name__ == "__main__":
    main()
