from matplotlib import pyplot as plt
from afm.misc.helper_functions import normalize
from scipy.signal import spectrogram

class Plots:
    @staticmethod
    def plot_spectrogram(audio, fs, vmin=None, vmax=None):
        frequencies, times, spectrum = spectrogram(audio, fs=fs)
        spectrum = normalize(spectrum)
        plt.imshow(spectrum, aspect='auto', origin='lower', extent=[times.min(), times.max(), frequencies.min(), frequencies.max()], vmin=vmin, vmax=vmax)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Spectrogram')
        plt.colorbar()
        plt.show()