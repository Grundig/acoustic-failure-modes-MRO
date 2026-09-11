from pathlib import Path

import numpy as np
from pydub import AudioSegment

class AudioLoader:
    def __init__(self):
        pass

    def load_audio(self, file_path: str | Path, mono: bool = True) -> tuple[np.ndarray, int]:
        """
        Load an audio file into a NumPy array.

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

        audio = AudioSegment.from_file(file_path)

        if mono:
            audio = audio.set_channels(1)

        sample_rate = audio.frame_rate
        sample_width = audio.sample_width
        channels = audio.channels

        samples = np.array(audio.get_array_of_samples())

        if channels > 1:
            samples = samples.reshape((-1, channels))

        max_possible_value = float(1 << (8 * sample_width - 1))
        samples = samples.astype(np.float32) / max_possible_value

        return samples, sample_rate


if __name__ == "__main__":
    from tkinter import filedialog
    loader = AudioLoader()
    audio_data, fs = loader.load_audio(filedialog.askopenfilename())

    print(f"Sample rate: {fs}")
    print(f"Audio shape: {audio_data.shape}")
    print(f"Audio dtype: {audio_data.dtype}")