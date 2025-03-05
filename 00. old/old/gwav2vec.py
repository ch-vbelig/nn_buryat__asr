import torch
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)

torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

import IPython
import matplotlib.pyplot as plt
from torchaudio.utils import download_asset

# SPEECH_FILE = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

print("Sample Rate:", bundle.sample_rate)

print("Labels:", bundle.get_labels())

model = bundle.get_model().to(device)

print(model.__class__)
print(model)


PATH = 'old/audio2/id_1_0001.wav'
waveform, sr = torchaudio.load(PATH)

waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate).to(device)

emissions, _ = model(waveform)

labels = bundle.get_labels()

class GreedyCTCDecoder(torch.nn.Module):
  def __init__(self, labels, blank=0):
    super().__init__()
    self.labels = labels
    self.blank = blank

  def forward(self, emission: torch.Tensor) -> str:
    """Given asdf sequence emission over labels, get the best path string
    Args:
      emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

    Returns:
      str: The resulting transcript
    """
    indices = torch.argmax(emission, dim=-1)  # [num_seq,]
    indices = torch.unique_consecutive(indices, dim=-1)
    indices = [i for i in indices if i != self.blank]
    return ''.join([self.labels[i] for i in indices])

decoder = GreedyCTCDecoder(labels=bundle.get_labels())
transcript = decoder(emissions[0])

print(transcript)

model.aux = torch.nn.Linear(in_features=768, out_features=61)

print(model)