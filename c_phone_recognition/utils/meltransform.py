import torch
import torchaudio
import librosa
import matplotlib.pyplot as plt
import c_phone_recognition.utils.config as config
import numpy as np

class MelTransform:
    def __init__(self,
                 sample_rate=config.SAMPLE_RATE,
                 n_fft=config.N_FFT,
                 win_length=config.WIN_LENGTH,
                 hop_length=config.HOP_LENGTH,
                 n_mels=config.N_MELS
                 ):
        """
        :param: sample_rate: (int): audio sample rate (default: 16kHz)
        :param: n_fft: (int): num of samples per frame (default: 1024 samples)
        :param: win_length: (int): window length (default: 512 samples)
        :param: hop_length: (int): overlap between frames (default: 512 samples)
        :param: n_mels: (int): num of mel filterbanks (default: 128 bins)
        """

        self.target_sample_rate = sample_rate

        self.spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_fft=n_fft,
            win_length=win_length,
            n_mels=n_mels
        )

        self.freq_masking = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=5
        )
        self.time_masking = torchaudio.transforms.TimeMasking(
            time_mask_param=5
        )

        self.amplitude_to_db_transform = torchaudio.transforms.AmplitudeToDB(
            stype='power',
            top_db=80
        )

    def build_waveform(self, audio_path, time_first=True):
        # signal: (n_channels, n_samples)
        signal, sr = torchaudio.load(audio_path)

        signal = self._resample_if_necessary(signal, sr)

        # signal: (n_channels, n_samples) -> (1, n_samples)
        signal = self._mix_down_if_necessary(signal)

        signal = signal.squeeze()

        return signal


    def build_spectrogram(self, audio_path, time_first=True):
        """
        :param audio_path: (str): path to audio file
        :param time_first: (bool): if True the ts will go first, used for padding in collate_fn
        :return: Log-scaled Mel spectrogram
        """
        # signal: (n_channels, n_samples)
        signal, sr = torchaudio.load(audio_path)

        signal = self._resample_if_necessary(signal, sr)

        # signal: (n_channels, n_samples) -> (1, n_samples)
        signal = self._mix_down_if_necessary(signal)

        # signal: (n_channels, n_mels, ts)
        signal = self.spec_transform(signal)

        for i in range(3):
            signal = self.freq_masking(signal)

        for i in range(10):
            signal = self.time_masking(signal)

        signal = self.amplitude_to_db_transform(signal)

        if time_first:
            # signal: (ts, n_channels, n_mels)
            signal = signal.permute(2, 0, 1)

        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    @staticmethod
    def plot_spectrogram(spectrogram, sr=config.SAMPLE_RATE, title=None, y_label="freq_bin", time_first=False):
        """
        :param spectrogram: (tensor): melspectrogram -> (1, n_mels, ts)
        :param sr: (int): sample rate of audio (defaul: 16kHz)
        :param title: (str): name of plot
        :param y_label: (str): label of y axes
        :param time_first: (bool): if True, melspectrogram comes as (ts, 1, n_mels)
        """
        spectrogram = spectrogram.squeeze()

        # convert into np.array
        spec = np.array(spectrogram)

        if time_first:
            spec = np.transpose(spec, (1, 0))

        fig, ax = plt.subplots()

        img = librosa.display.specshow(
            spec,
            x_axis="time",
            y_axis="mel",
            sr=sr,
            ax=ax,
            cmap='Reds'
        )

        if title:
            ax.set_title(title)

        ax.set_ylabel(y_label)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        plt.show()