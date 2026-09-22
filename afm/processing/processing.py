import torchaudio
from afm.machine_learning.embedding import Embedding, BaselineEmbeddings

class Processor:
    def __init__(self):
        self.embedding = Embedding()
        self.baseline = None

    @staticmethod
    def get_spectrum(audio, n_fft = 1028):
        transform = torchaudio.transforms.Spectrogram(n_fft)
        return transform(audio)

    @staticmethod
    def get_log_mel_spectrum(audio, sample_rate, n_fft = 1028, n_mels = 128):
        transform = torchaudio.transforms.MelSpectrogram(sample_rate, n_fft)
        return transform(audio)

    @staticmethod
    def get_pitch(audio, sample_rate):
        return torchaudio.functional.detect_pitch_frequency(audio, sample_rate)

    def get_features(self, waveform):
        return self.embedding.process_extract_features(waveform)

    def get_baseline_features(self, waveform):
        self.baseline = BaselineEmbeddings(window_s=10, sample_rate=waveform[0][1])
        baseline_bank = self.baseline.build([w[0] for w in waveform])

        return baseline_bank

    def get_comparison_score(self, audio, baseline):
        return self.baseline.score(audio, baseline)