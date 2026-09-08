from pathlib import Path

import numpy as np
from pydub import AudioSegment


def load_audio(file_path: str | Path, mono: bool = True) -> tuple[np.ndarray, int]:
    """
    Load an audio file into a NumPy array.

    Supports formats such as .wav, .aac, .m4a, .mp3, .flac, etc.,
    as long as ffmpeg can decode them.

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
    audio_data, sr = load_audio("example.m4a")

    print(f"Sample rate: {sr}")
    print(f"Audio shape: {audio_data.shape}")
    print(f"Audio dtype: {audio_data.dtype}")