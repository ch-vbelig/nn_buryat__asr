import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from cmodel import PhoneRecognitionModel, PhoneRecognitionModelWav2Vec
import utils.config as config
import json
from bdataset import SpeechDataset, SpeechDatasetWav2Vec
import dengine as engine
import torch.optim.lr_scheduler as lr_scheduler
from pathlib import Path


MODEL_PATH = '../models/phone_model_wav2vec.pth'
LABEL_PATH = '../../c_phone_recognition/data/bur_phone_set.txt'

torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PhoneRecognitionModelWav2Vec(61)
model = engine.load_model(model, MODEL_PATH).to(device)


PATH = 'old/test_audio/test3.wav'
waveform, sr = torchaudio.load(PATH)

waveform = torchaudio.functional.resample(waveform, sr, 16000).to(device)

emissions = model(waveform)
print(emissions.size())

def get_labels():
  labels = ['<SIL>', '<SPACE>']
  with open(LABEL_PATH) as fp:
    lines = fp.readlines()
  lines = [line.strip() for line in lines]
  labels.extend(lines)
  labels.append('<BLANK>')
  print(len(labels))
  print(labels)

  return labels


labels = get_labels()

class GreedyCTCDecoder(torch.nn.Module):
  def __init__(self, labels, blank=60):
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
    return ' '.join([self.labels[i] for i in indices])

decoder = GreedyCTCDecoder(labels=labels)
transcript = decoder(emissions[0])

print(transcript)