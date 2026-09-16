from matplotlib import pyplot as plt
import librosa
import torchaudio
import torch

class Plots:

    @staticmethod
    def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
        specgram = specgram.squeeze(0).numpy()
        if ax is None:
            _, ax = plt.subplots(1, 1)
        if title is not None:
            ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")

    @staticmethod
    def plot_waveform(waveform, sr, title="Waveform", ax=None):
        waveform = waveform.numpy()

        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / sr

        if ax is None:
            _, ax = plt.subplots(num_channels, 1)

        ax.plot(time_axis, waveform[0], linewidth=1)
        ax.grid(True)
        ax.set_xlim([0, time_axis[-1]])
        ax.set_title(title)