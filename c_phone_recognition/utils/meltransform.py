import torch
import torchaudio
from pathlib import Path
import librosa
import matplotlib.pyplot as plt
import pandas as pd
import c_phone_recognition.utils.config as config
import numpy as np

class MelTransform:
    def __init__(self,
                 audio_dir,
                 data_csv,
                 sample_rate=config.SAMPLE_RATE,
                 n_fft=config.N_FFT,
                 win_length=config.WIN_LENGTH,
                 hop_length=config.HOP_LENGTH,
                 n_mels=config.N_MELS,
                 max_duration=config.MAX_DURATION,
                 ):
        """
        :param: audio_dir (str): directory containing *.wav files
        :param: data_csv (str): phrase-to-phones *.csv file -> Table with | phrase | phones | file_name |
        :param: max_duration (str): phrase-to-phones *.csv file -> Table with | phrase | phones | file_name |
        :param: sample_rate (int): the target sample rate -> default: 16 kHz
        :param: n_fft (int): num of samples per frame -> default: 1024 samples
        :param: hop_length (int): overlap between frames -> default: 512 samples
        :param: n_mels (int): num of mel filterbanks -> default: 128 bins

        :attr: self.df (pd.DataFrame): dataframe with columns | phrase | phones | file_name |
        """

        self.audio_dir = audio_dir
        self.data_csv_path = data_csv

        self.target_sample_rate = sample_rate
        self.max_duration = max_duration

        self.spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            # normalized=True
        )

        self.freq_masking = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=7
        )
        self.time_masking = torchaudio.transforms.TimeMasking(
            time_mask_param=15
        )

        self.amplitude_to_db_transform = torchaudio.transforms.AmplitudeToDB(
            stype='power',
            top_db=80
        )

        self.df = pd.read_csv(data_csv, header=None, index_col=0).dropna(how='any')
        self.df.columns = ['speaker_id', 'phrase', 'phones', 'file_name']

        # get all audio_dir/*.wav files
        self.audio_paths = self._search_files()

    def build_spectrograms(self, n_files=None):
        """
        :return: inputs (list of tensors): list of melspectrograms (tensors) -> (n, ts, 1, n_mels)
        :return: phone_representations (list of str): list of phone representations (not padded) -> (n, )
        :return: input_lengths (list of int): melspectrogram time steps before padding -> (n, )
        """
        inputs = []
        phone_representations = []
        input_lengths = []

        if n_files is None:
            n_files = len(self.audio_paths)

        for audio_path in self.audio_paths[:n_files]:
            # spectrogram: (1, 128, ts)
            spectrogram, input_length, file_stem = self._build_spectrogram(audio_path)
            # permute to bring ts to the first dim -> required for torch.nn.utils.rnn.pad_sequence
            spectrogram = spectrogram.permute(2, 0, 1)  # (ts, 1, 128)

            phone_repr = self.df[self.df['file_name'] == file_stem]['phones'].values[0]

            # inputs: (n, ts, 1, n_mels)
            inputs.append(spectrogram)

            # phone_representations: (n, ts)
            phone_representations.append(phone_repr)

            input_lengths.append(input_length)
        return inputs, phone_representations, input_lengths

    def _search_files(self):
        paths = []
        for stem in self.df['file_name']:
            path = Path(self.audio_dir) / f'{stem}.wav'
            paths.append(path)
        return paths

    def _build_spectrogram(self, audio_path):
        # signal: (n_channels, n_samples)
        signal, sr = torchaudio.load(audio_path)
        label = audio_path.stem

        if signal.size(1) // sr > self.max_duration:
            raise RuntimeError(f"File {audio_path.name} is longer than {self.max_duration} seconds.")

        signal = self._resample_if_necessary(signal, sr)

        # signal: (n_channels, n_samples) -> (1, n_samples)
        signal = self._mix_down_if_necessary(signal)

        # signal: (n_channels, n_mels, ts) : (1, 128, ts)
        signal = self.spec_transform(signal)


        # for i in range(7):
        #     signal = self.freq_masking(signal)
        #
        # for i in range(20):
        #     signal = self.time_masking(signal)

        signal = self.amplitude_to_db_transform(signal)

        signal_length = signal.size(2)

        return signal, signal_length, label

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def plot_spectrogram(self, spectrogram, sr=config.SAMPLE_RATE, title=None, y_label="freq_bin", ts_first=False,):
        """
        :param spectrogram (tensor): melspectrogram -> (1, n_mels, ts)
        :param ts_first (bool): if True, melspectrogram comes as (ts, 1, n_mels)
        """
        spectrogram = spectrogram.squeeze()

        # convert into np.array
        spec = np.array(spectrogram)

        if ts_first:
            spec = np.transpose(spec, (1, 0))

        fig, ax = plt.subplots()

        img = librosa.display.specshow(
            spec,
            x_axis="time",
            y_axis="mel",
            sr=sr,
            fmax=8000,
            ax=ax
        )

        if title:
            ax.set_title(title)
        ax.set_ylabel(y_label)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        plt.show()


if __name__ == "__main__":
    AUDIO_DIR = '../data/audio'
    DATA_PATH = '../data/word_to_phones_dict.csv'
    SAMPLE_RATE = 16000

    mel_transform = MelTransform(AUDIO_DIR, DATA_PATH, sample_rate=SAMPLE_RATE)
    specs, fnames, targets = mel_transform.build_spectrograms(2)
    print(targets)
