import re
from pathlib import Path
import re

import torchaudio
import torch
from pathlib import Path

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
labels = bundle.get_labels()

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model = bundle.get_model().to(device)

AUDIO_DIR = '../old/lexicon_words/'
LEXICON_FILE = 'data/bur_lexicon.txt'
LEXICON_FILE_SAVE = 'data/bur_lexicon_2.txt'


def get_lines(fpath):
    with open(LEXICON_FILE, encoding='UTF-8') as fp:
        lines = fp.read().split('\n')
        lines = [line.strip() for line in lines]

    return lines


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
    return transcript.replace(' | ', ' ')


lines = get_lines(AUDIO_DIR)
words = [line.split()[0] for line in lines]
audio_files = list(Path(AUDIO_DIR).glob('*.wav'))
pairs = []

for i in range(0, len(audio_files)):
    pattern = f'Voice {str(i+1).rjust(3, '0')}'

    file = [fpath for fpath in audio_files if fpath.stem == pattern][0]
    # print(file)

    predicted = predict(file)
    pair = f'{words[i]}\t{predicted}'
    pairs.append(pair)


with open(LEXICON_FILE_SAVE, 'w', encoding='UTF-8') as fp:
    text = '\n'.join(pairs)
    fp.write(text)