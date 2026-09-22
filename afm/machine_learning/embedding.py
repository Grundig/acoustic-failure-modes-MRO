from speechbrain.lobes.models.beats import BEATs, BEATsConfig
from pathlib import Path
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

DEVICE = torch.device("cpu")

class Embedding:
    def __init__(self):
        self.cpt = Path(__file__).resolve().parent / Path("../machine_learning/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt")

        self.model = BEATs(self.cpt, freeze=True)
        self.model.to(DEVICE)
        self.model.eval()

    def process_extract_features(self, audio):
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio_lengths = torch.ones(audio.shape[0], device=audio.device)
        return self.model.extract_features(audio, audio_lengths)

    def process_forward(self, audio):
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio_lengths = torch.ones(audio.shape[0], device=audio.device)
        return self.model.forward(audio, audio_lengths)

class BaselineEmbeddings:
    def __init__(self,  window_s=10, sample_rate=16000):
        self.cpt = Path(__file__).resolve().parent / Path(
            "../machine_learning/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt")

        self.model = BEATs(self.cpt, freeze=True)
        self.model.to(DEVICE)
        self.model.eval()

        self.window_samples = window_s * sample_rate

    def _embed(self, audio):
        # audio: (1, T) mono, already resampled to model's expected rate
        audio_lengths = torch.ones(1, device=audio.device)
        with torch.no_grad():
            feats = self.model.extract_features(audio, audio_lengths)  # (1, time, dim)
        return feats[0].mean(dim=1).squeeze(0).numpy()  # (dim,) mean-pooled

    def _chunk(self, audio):
        # split into fixed windows,
        n = audio.shape[-1] // self.window_samples
        return [audio[..., i * self.window_samples:(i + 1) * self.window_samples] for i in range(n)]

    def build(self, normal_recordings):
        """normal_recordings: list of (1, T) waveforms captured under known-good conditions"""
        embs = []
        for audio in normal_recordings:
            # if audio.dim() == 1:
            #     audio = audio.unsqueeze(0)
            for chunk in self._chunk(audio):
                embs.append(self._embed(chunk))
        return np.stack(embs)  # your reference bank

    def score(self, audio, baseline, k=5):
        """Anomaly score for a new recording: mean distance to k nearest baseline embeddings"""
        nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(baseline)
        scores = []
        for chunk in self._chunk(audio):
            emb = self._embed(chunk).reshape(1, -1)
            dist, _ = nn.kneighbors(emb)
            scores.append(dist.mean())
        return np.array(scores)  # one score per window; higher = more anomalous