from speechbrain.lobes.models.beats import BEATs, BEATsConfig
from pathlib import Path
import torch

DEVICE = torch.device("cpu")

class Embedding:
    def __init__(self):
        self.cpt = Path(__file__).resolve().parent / Path("../machine_learning/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt")

        self.model = BEATs(self.cpt, freeze=True)
        self.model.to(DEVICE)
        self.model.eval()

    def process(self, audio):
        # audio = torch.randn(4, 10000)  # Batch of 4 audio signals
        # wav_lengths = torch.tensor([1.0, 0.5, 0.75, 1.0])
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        wav_lengths = torch.ones(audio.shape[0], device=audio.device)
        return self.model.extract_features(audio, wav_lengths)