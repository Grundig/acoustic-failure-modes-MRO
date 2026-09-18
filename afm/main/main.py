from matplotlib import pyplot as plt

from afm.importers.reading_data import AudioLoader
from afm.processing.processing import Processor
from afm.visualisation.plots import Plots
from afm.machine_learning.embedding import Embedding
from tkinter import filedialog

import torch
torch.set_num_threads(4)
torch.no_grad()

class Main:
    def __init__(self):
        self.loader = AudioLoader()
        self.processor = Processor()
        self.plots = Plots()
        self.embedding = Embedding()

        self.filepath = filedialog.askopenfilename()
        self.waveform = None
        self.sample_rate = None

    def load(self):
        self.waveform, self.sample_rate = self.loader.load_audio(self.filepath)

    def run(self):
        self.load()
        log_mel = self.processor.get_log_mel_spectrum(self.waveform, self.sample_rate)
        spectr = self.processor.get_spectrum(self.waveform)
        pitch = self.processor.get_pitch(self.waveform, self.sample_rate)

        fig, ax = plt.subplots(3,1)
        self.plots.plot_waveform(self.waveform, self.sample_rate, ax=ax[0])
        self.plots.plot_spectrogram(log_mel, ax=ax[1], title="Log-mel Spectrogram")
        self.plots.plot_spectrogram(spectr, ax=ax[2], title="Spectrogram")
        self.plots.plot_pitch(self.waveform, self.sample_rate, pitch)
        fig.tight_layout()
        plt.show()

        extracted_features = self.embedding.process(self.waveform)
        print(extracted_features)

if __name__ == "__main__":
    main = Main()
    main.run()