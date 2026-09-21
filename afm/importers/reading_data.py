from pathlib import Path
import torch
import torchaudio

class AudioLoader:
    def __init__(self):
        pass

    @staticmethod
    def load_audio(path: str | Path, mono: bool = True, num_frames: int = -1, file_extension: str = ".wav") -> list[tuple[torch.Tensor, int]]:
        """
        Args:
            path: Path to the audio file or directory.
            mono: If True, convert audio to mono.
            num_frames: Number of frames to load.
            file_extension: If path is a directory, load files with this extension.

        Returns:
            A list of tuples, each containing:
                - audio_samples: Tensor of samples
                - sample_rate: Sampling rate of the audio file
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Audio file or directory not found: {path}")

        audio_list = []
        if path.is_file():
            files = [path]
        elif path.is_dir():
            files = sorted(list(path.glob(f"*{file_extension}")))
        else:
            raise ValueError(f"Path is neither a file nor a directory: {path}")

        for file_path in files:
            waveform, sample_rate = torchaudio.load(file_path, num_frames=num_frames)
            if mono and waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            audio_list.append((waveform, sample_rate))

        return audio_list


if __name__ == "__main__":
    from tkinter import filedialog
    loader = AudioLoader()
    audio_list = loader.load_audio(filedialog.askopenfilename())

    for audio_data, fs in audio_list:
        print(f"Sample rate: {fs}")
        print(f"Audio shape: {audio_data.shape}")
        print(f"Audio dtype: {audio_data.dtype}")