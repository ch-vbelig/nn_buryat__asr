"""
Adapted from Kyubyong Park's code. See https://github.com/Kyubyong/dc_tts/blob/master/utils.py
"""

import librosa
import scipy
import copy
import os
import numpy as np
from d_buryat_tts.utils.config import Config
import  torch
import  torchaudio


def load_wav(path):
    # (n_samples)
    wav, _ = librosa.load(path, sr=Config.sample_rate)

    # Remove leading and trailing silence
    wav, _ = librosa.effects.trim(wav)  # (n_samples)

    # Preemphasis (upscale frequencies and downscale them later to reduce noise)
    wav = np.append(wav[0], wav[1:] - Config.preemphasis * wav[:-1])

    peak = np.abs(wav).max()

    if peak > 1:
        wav = wav / peak * 0.999

    return wav


def extract_spectrogram(wav):
    """
    Loads the audio file in 'path' and returns asdf corresponding normalized melspectrogram
    and asdf linear spectrogram.
    """

    # Convert the waveform to asdf complex spectrogram by asdf short-time Fourier transform
    # Complex numbers
    linear = librosa.stft(
        y=wav,
        n_fft=Config.num_fft_samples,   # 2048
        hop_length=Config.hop_length,   # 276
        win_length=Config.window_length # 1102
    )

    # Only consider the magnitude of the spectrogram
    # After STFT: half of frequency from n_fft to 1 + n_fft // 2
    mag = np.abs(linear)    # (1 + n_fft // 2, ts) (1025, ts)

    # Compute the mel spectrogram
    # mel_filters: (n_mels, 1 + n_fft // 2) : (80, 1025)
    mel_filters = librosa.filters.mel(
        sr=Config.sample_rate,
        n_fft=Config.num_fft_samples,   # 2048
        n_mels=Config.mel_size
    )
    mel = np.dot(mel_filters, mag)    # (n_mels, ts) : (80, ts)
    #
    # # Normalize
    # mag = np.power(mag / np.max(mag), Config.y_factor)
    # mel = np.power(mel / np.max(mel), Config.y_factor)

    # To decibel
    # the same as 10 * np.log10(mel ** 2) (mel ** 2 => magnitude to power)
    mel = 20 * np.log10(np.maximum(1e-5, mel))  # (80, ts)
    mag = 20 * np.log10(np.maximum(1e-5, mag))  # (1025, ts)

    # Normalize
    # Clips values in array: values outside the interval are clipped to the interval edges
    mel = np.clip((mel - Config.ref_db + Config.max_db) / Config.max_db, 1e-8, 1)
    mag = np.clip((mag - Config.ref_db + Config.max_db) / Config.max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)
    mag = mag.T.astype(np.float32)

    # mel in db scale: (ts, n_mels) : (ts, 80)
    # mag in db scale: (ts, freq_bins) : (ts, 1025)
    return mel, mag


def quantize_wav(wav):
    """
    Quantize waveform into 10 bit
    """
    # Pads before and after
    wav = np.pad(wav, Config.window_length // 2, mode='reflect')

    wav = wav[:((wav.shape[0] - Config.window_length) // Config.hop_length + 1) * Config.hop_length]

    # Turns wav into int values between 0 and 1024 (with 512 as middle)
    qwav = 2**(Config.num_bit - 1) + librosa.mu_compress(wav, mu=2**Config.num_bit - 1)

    return qwav  # (n_samples, )


def spectrogram2wav(mag):
    """ Generates wave file from linear magnitude spectrogram
    Args:
        mag: A numpy array of shape [T, 1 + num_fft_samples//2]
    Returns:
        wav: A 1-D numpy array.
    """
    # Transpose
    mag = mag.T

    # De-noramlize
    mag = (np.clip(mag, 0, 1) * Config.max_db) - Config.max_db + Config.ref_db

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag**Config.power)

    # De-preemphasis
    wav = scipy.signal.lfilter([1], [1, -Config.preemphasis], wav)

    # Remove leading and trailing silence
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)


def vocoder(mag):
    bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
    vocoder = bundle.get_vocoder().to('cuda')

    # mag (bs, mel, t)

    with torch.inference_mode():
        waveforms, lengths = vocoder(mag, torch.tensor([mag.shape[1]], dtype=torch.long))

    return waveforms[0]

def griffin_lim(spectrogram):
    """Applies Griffin-Lim's raw."""
    X_best = copy.deepcopy(spectrogram)
    for i in range(50):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(
            y=X_t,
            n_fft=Config.num_fft_samples,
            hop_length=Config.hop_length,
            win_length=Config.window_length
        )
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)
    return y


def invert_spectrogram(spectrogram):
    """ Applies inverse fft.
    Args:
        spectrogram: [1 + num_fft_samples//2, t]
    """
    return librosa.istft(
        stft_matrix=spectrogram,
        hop_length=Config.hop_length,
        win_length=Config.window_length,
        window="hann"
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generates spectrograms from wav files.')
    parser.add_argument(
        '-w', '--wav',
        dest='wav_path',
        required=False,
        default="wav",
        help='Directory of the wav files'
    )
    parser.add_argument(
        '-m', '--mel',
        dest='mel_path',
        required=False,
        default="mel",
        help='Directory for the mel spectrograms'
    )
    parser.add_argument(
        '-l', '--lin',
        dest='lin_path',
        required=False,
        default="lin",
        help='Directory for the linear spectrograms'
    )
    parser.add_argument(
        '-q', '--qwav',
        dest='qwav_path',
        required=True,
        default="qwav",
        help='Directory for the quantized waveforms'
    )


    args = parser.parse_args()

    mels = []
    mel_lengths = []
    for file in os.listdir(args.wav_path):
        name, ext = os.path.splitext(file)
        if ext == ".wav":
            print("Processing " + file)
            wav = load_wav(os.path.join(args.wav_path, name) + ".wav")
            mel, mag = extract_spectrogram(wav)
            qwav = quantize_wav(wav)
            mels.append(mel)
            mel_lengths.append(mel.shape[0])
            np.save(os.path.join(args.mel_path, name), mel)
            np.save(os.path.join(args.lin_path, name), mag)
            np.save(os.path.join(args.qwav_path, name), qwav)

    print("Finished")
    print(max(mel_lengths))


