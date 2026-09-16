import torchaudio

class Processor:
    def __init__(self):
        pass

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
