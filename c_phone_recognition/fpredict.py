import torch
import torchaudio
from pathlib import Path
import dengine as engine
import utils.config as config
from cmodel import PhoneRecognitionModel
from c_phone_recognition.utils.old import PhoneRecognitionModelBasic
from utils.meltransform import MelTransform
from utils.converter import PhoneConverter
from torchaudio.models.decoder import ctc_decoder

PHONE_SET_PATH = './data/bur_phone_set.txt'
MODEL_PATH = 'models/phone_model_normalized_2d_conv_batchnorm.pth'
AUDIO_PATH = Path('./data/test_audio/test.wav')



def build_spectrogram(audio_path, transform, amplitude_to_db_transform):

    # signal: (n_channels, n_samples)
    signal, sr = torchaudio.load(audio_path)
    label = audio_path.stem

    signal = resample_if_necessary(signal, sr)

    # signal: (n_channels, n_samples) -> (1, n_samples)
    signal = mix_down_if_necessary(signal)
    signal = pad_if_necessary(signal, label)

    # signal: (n_channels, n_mels, ts) : (1, 128, ts)
    signal = transform(signal)
    signal = amplitude_to_db_transform(signal)

    mean = -9.52
    std = 11.72

    signal = (signal - mean) / std
    return signal


def resample_if_necessary(signal, sr):
    if sr != config.SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, config.SAMPLE_RATE)
        signal = resampler(signal)
    return signal


def mix_down_if_necessary(signal):
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal


def pad_if_necessary(signal, label):
    seq_length = config.MAX_DURATION * config.SAMPLE_RATE
    padded = torch.zeros((1, seq_length))
    end_idx = signal.size(1)
    if seq_length < end_idx:
        raise RuntimeError(f"File {label} is longer than {config.MAX_DURATION} seconds.")
    padded[0][:end_idx] = signal[0]
    return padded

def convert_data(ids, converter):
    _tokens = [converter.index_to_phone[i] for i in ids if i in converter.index_to_phone]

    tokens = []
    for t in _tokens:
        if len(tokens) > 0 and tokens[-1] == t:
            continue
        tokens.append(t)
    return tokens


def beam_search_ctc_decoder(emission):
    LEXICON_FILE = "decoder/bur_word_to_phones_dict.txt"
    TOKENS_FILE = "decoder/bur_phone_set.txt"

    beam_search_decoder = ctc_decoder(
        lexicon=LEXICON_FILE,
        tokens=TOKENS_FILE,
        lm=None,
        nbest=3,
        beam_size=1500,
        word_score=-0.2,
        blank_token='<BLANK>',
        sil_token='<SIL>',

    )
    emission = emission.permute(1, 0, 2)
    beam_search_result = beam_search_decoder(emission)
    beam_search_tokens = " ".join(beam_search_decoder.idxs_to_tokens(beam_search_result[0][0].tokens))
    beam_search_transcript = " ".join(beam_search_result[0][0].words).strip()
    print(beam_search_tokens)
    print(beam_search_transcript)

if __name__ == '__main__':
    # load and set to eval model
    model = PhoneRecognitionModel(61)
    model = engine.load_model(model, MODEL_PATH)
    model.eval()

    transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS
        )

    amplitude_to_db_transform = torchaudio.transforms.AmplitudeToDB(
        stype='amplitude',
        top_db=80
    )

    spectrogram = build_spectrogram(AUDIO_PATH, transform, amplitude_to_db_transform)

    print(spectrogram.size())

    data = spectrogram.unsqueeze(0)

    log_probs = model(data)

    beam_search_ctc_decoder(log_probs)

    _, ids = torch.max(log_probs, dim=-1)
    ids = ids.squeeze(1).numpy().tolist()

    phone_converter = PhoneConverter(PHONE_SET_PATH)

    res = convert_data(ids, phone_converter)
    print(res)