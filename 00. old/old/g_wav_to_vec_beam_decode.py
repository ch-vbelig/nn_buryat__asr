import torch
import torchaudio
from pathlib import Path
from torchaudio.models.decoder import ctc_decoder

AUDIO_DIR = 'temp'
TEXT_DIR = 'transcript'
LEXICON_FILE = "../phone_lexicon_2.txt"

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
TOKENS = bundle.get_labels()

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model = bundle.get_model().to(device)

beam_search_decoder = ctc_decoder(
    lexicon=LEXICON_FILE,
    tokens=TOKENS,
    lm=None,
    nbest=5,
    beam_size=1500,
    word_score=-0.8,
    blank_token='-',
)

def beam_search_ctc_decoder(emission, labels):
    # emission : (B, T, N)
    emission = emission.to('cpu')
    beam_search_result = beam_search_decoder(emission)
    beam_search_tokens = " ".join(beam_search_decoder.idxs_to_tokens(beam_search_result[0][0].tokens))
    beam_search_transcript = " ".join(beam_search_result[0][0].words).strip()

    return beam_search_transcript.replace('_', ' ')

def predict(fpath):
    waveform, sample_rate = torchaudio.load(fpath)
    waveform = waveform.to(device)

    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

    with torch.inference_mode():
        emission, _ = model(waveform)

    return emission

def save_text(fpath, text_dir, text):
    text_path = Path(text_dir) / f'{fpath.stem}.txt'

    with open(text_path, 'w', encoding='UTF-8') as fp:
        fp.write(text)

audio_files = list(Path(AUDIO_DIR).glob('*.wav'))

for fpath in audio_files:
    emission = predict(fpath)
    text = beam_search_ctc_decoder(emission, TOKENS)
    print(text)
    save_text(fpath, TEXT_DIR, text)