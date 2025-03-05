import re

import torchaudio
import torch
from pathlib import Path

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
print('Sample rate', bundle.sample_rate)
print('Labels', bundle.get_labels())
labels = bundle.get_labels()

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model = bundle.get_model().to(device)

AUDIO_DIR = 'temp'
TEXT_DIR = '../text_from_wav_2_vec'
audio_files = list(Path(AUDIO_DIR).glob('*.wav'))


def predict(speech_file):
    waveform, sample_rate = torchaudio.load(speech_file)
    waveform = waveform.to(device)

    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

    with torch.inference_mode():
        emission, _ = model(waveform)

    ids = torch.argmax(emission[0], dim=-1)
    ids = torch.unique_consecutive(ids, dim=-1)
    ids = [i for i in ids if i != 0]
    transcript = " ".join([labels[i] for i in ids])
    return transcript.replace('|', '<SPACE>')


def save_text(fpath, text_dir, text):
    text_path = Path(text_dir) / f'{fpath.stem}.txt'

    with open(text_path, 'w') as fp:
        fp.write(text)

def clean_text(text):

    text = re.sub(r'W', '', text)
    text = re.sub('  ', ' ', text)
    text = re.sub('<SPACE>', '_', text)
    text = re.sub('E', 'EH', text)
    text = re.sub('Y', 'J', text)

    return text


for fpath in audio_files:
    text = predict(fpath)
    # text = clean_text(text)
    print(text)
    # save_text(fpath, TEXT_DIR, text)

