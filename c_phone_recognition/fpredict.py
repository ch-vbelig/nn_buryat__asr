import torch
import torchaudio
from pathlib import Path
import dengine as engine
import utils.config as config
from cmodel import PhoneRecognitionModel
from utils.meltransform import MelTransform
from utils.converter import PhoneConverter

PHONE_SET_PATH = './data/bur_phone_set.txt'
MODEL_PATH = './models/phone_model_normalized2.pth'
AUDIO_PATH = Path('./data/audio/id_0_1022.wav')


def build_spectrogram(audio_path, transform, amplitude_to_db_transform):

    # signal: (n_channels, n_samples)
    signal, sr = torchaudio.load(audio_path)
    label = audio_path.stem
    signal = resample_if_necessary(signal, sr)
    # signal: (n_channels, n_samples) -> (1, n_samples)
    signal = mix_down_if_necessary(signal)
    signal = pad_if_necessary(signal, label)
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
        raise Exception(f"File {label} is longer than {config.MAX_DURATION} seconds.")
    padded[0][:end_idx] = signal[0]
    return padded

def convert_data(ids, converter):
    _tokens = []

    for i in ids:
        if i in converter.index_to_phone:
            token = converter.index_to_phone[i]
        else:
            # token = '`'
            continue
        _tokens.append(token)
    tokens = []

    for t in _tokens:
        if t in ['`']:
            continue
        if len(tokens) > 0 and tokens[-1] == t:
            continue
        tokens.append(t)
    return tokens

if __name__ == '__main__':
    model = PhoneRecognitionModel(60)
    model = engine.load_model(model, MODEL_PATH)

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

    _, ids = torch.max(log_probs, dim=-1)
    ids = ids.squeeze(1).numpy().tolist()


    phone_converter = PhoneConverter(PHONE_SET_PATH)

    res = convert_data(ids, phone_converter)
    print(res)



