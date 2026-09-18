from pathlib import Path
import torch
import torchaudio

class AudioLoader:
    def __init__(self):
        pass

    def load_audio(self, file_path: str | Path, mono: bool = True):
        """
        Args:
            file_path: Path to the audio file.
            mono: If True, convert audio to mono.

        Returns:
            A tuple containing:
                - audio_samples: NumPy array of normalized float32 samples in range [-1.0, 1.0]
                - sample_rate: Sampling rate of the audio file
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        waveform, sample_rate = torchaudio.load(file_path, num_frames=10000)
        if mono and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        return waveform, sample_rate


if __name__ == "__main__":
    from tkinter import filedialog
    loader = AudioLoader()
    audio_data, fs = loader.load_audio(filedialog.askopenfilename())

    print(f"Sample rate: {fs}")
    print(f"Audio shape: {audio_data.shape}")
    print(f"Audio dtype: {audio_data.dtype}")