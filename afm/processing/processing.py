from tkinter import filedialog
from afm.importers.reading_data import AudioLoader
from afm.visualisation.plots import Plots

class Processor:
    def __init__(self):
        pass

    @staticmethod
    def load():
        loader = AudioLoader()
        audio, sample_rate = loader.load_audio(filedialog.askopenfilename())

        return audio, sample_rate


    def main(self):
        audio, sample_rate = self.load()
        Plots.plot_spectrogram(audio, sample_rate, vmax=0.01)


if __name__ == "__main__":
    p = Processor()
    p.main()
