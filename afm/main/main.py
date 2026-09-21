from matplotlib import pyplot as plt

from afm.importers.reading_data import AudioLoader
from afm.processing.processing import Processor
from afm.visualisation.plots import Plots
from tkinter import filedialog

import torch
torch.set_num_threads(4)
torch.no_grad()

class Main:
    def __init__(self):
        self.loader = AudioLoader()
        self.processor = Processor()
        self.plots = Plots()

        self.filepath = filedialog.askdirectory()
        self.waveform = None
        self.sample_rate = None
        self.load()

    def load(self):
        audio_list = self.loader.load_audio(self.filepath, num_frames=-1)
        if audio_list:
            self.waveform, self.sample_rate = audio_list[0]
        else:
            raise ValueError(f"No audio files found in: {self.filepath}")

    def visualise(self):
        log_mel = self.processor.get_log_mel_spectrum(self.waveform, self.sample_rate)
        spectr = self.processor.get_spectrum(self.waveform)
        pitch = self.processor.get_pitch(self.waveform, self.sample_rate)

        fig, ax = plt.subplots(3, 1)
        self.plots.plot_waveform(self.waveform, self.sample_rate, ax=ax[0])
        self.plots.plot_spectrogram(log_mel, ax=ax[1], title="Log-mel Spectrogram")
        self.plots.plot_spectrogram(spectr, ax=ax[2], title="Spectrogram")
        self.plots.plot_pitch(self.waveform, self.sample_rate, pitch)
        fig.tight_layout()
        plt.show()

    def run(self):
        pass



if __name__ == "__main__":
    main = Main()
    main.run()