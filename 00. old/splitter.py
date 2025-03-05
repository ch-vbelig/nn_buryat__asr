import librosa
import torch
import torchaudio
import config
from pathlib import Path
import numpy as np


def open_audio_file(path):
    path = Path(path)
    x, sr = torchaudio.load(path)

    def _resample_if_necessary(signal, sample_rate):
        if sample_rate != config.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sample_rate, config.SAMPLE_RATE)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    x = _resample_if_necessary(x, sr)
    x = _mix_down_if_necessary(x)

    return x, path.stem

def save_file(signal, path):
    torchaudio.save(uri=path, src=signal, sample_rate=config.SAMPLE_RATE)
    print('saved in', path)

def split(signal):
    signal = signal.squeeze()
    duration = 10
    step = duration * config.SAMPLE_RATE

    segments = []
    for i in range(0, signal.size(0), step):

        segment = signal[i: i + step]
        segment = segment.unsqueeze(0)
        print(segment.size())
        segments.append(segment)

    return segments


def create_txt_file(fname, phones_dir):
    path = Path(phones_dir) / fname
    open(path, 'w').close()



def run_split(audio_file, audio_dir, phones_dir):
    # signal: (1, ts)
    signal, audio_id = open_audio_file(audio_file)

    segments = split(signal)

    dir_path = Path(audio_dir)

    for i, seg in enumerate(segments):
        save_file(segments[i], dir_path / f'{audio_id}_{i}.wav')
        create_txt_file(f'{audio_id}_{i}.txt', phones_dir)


def load_spectrogram(path):
    sample_rate = 22050
    num_fft_samples = 2048
    hop_length = 276
    win_length = 1102
    mel_size = 80

    y, _ = librosa.load(path, sr=22050)

    # Remove leading and trailing silence
    y, _ = librosa.effects.trim(y)

    # Preemphasis (upscale frequencies and downscale them later to reduce noise)
    y = np.append(y[0], y[1:] - .97 * y[:-1])

    # Convert the waveform to asdf complex spectrogram by asdf short-time Fourier transform
    linear = librosa.stft(y=y, n_fft=2048, hop_length=276,
                          win_length=1102)

    # Only consider the magnitude of the spectrogram
    mag = np.abs(linear)

    # Compute the mel spectrogram
    mel_basis = librosa.filters.mel(
        sr=sample_rate,
        n_fft=num_fft_samples,
        n_mels=mel_size)
    mel = np.dot(mel_basis, mag)

    print(mel.shape)


if __name__ == '__main__':
    AUDIO_FILE = "old/temp/broadcast_1_0.wav"
    AUDIO_DIR = 'old/temp'
    PHONES_DIR = 'old/temp_phones'

    # run_split(AUDIO_FILE, AUDIO_DIR, PHONES_DIR)
    load_spectrogram(AUDIO_FILE)
