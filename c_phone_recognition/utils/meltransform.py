import torch
import torchaudio
from pathlib import Path
import librosa
import matplotlib.pyplot as plt
import pandas as pd

class MelTransform:
    def __init__(self,
                 audio_dir,
                 data_csv,
                 sample_rate,
                 n_fft,
                 hop_length,
                 n_mels,
                 max_s_length=10
                 ):
        """
        :param: audio_dir: directory containing *.wav files
        transformation params: sr = 16 kHz, n_fft = 1024 frames, hop_length = 512 frames, n_mels = 128 bins
        """

        self.audio_dir = audio_dir
        self.data_txt_path = data_csv
        self.target_sample_rate = sample_rate
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            # normalized=True
        )
        self.amplitude_to_db_transform = torchaudio.transforms.AmplitudeToDB(
            stype='amplitude',
            top_db=80
        )
        self.df = pd.read_csv(data_csv, header=None, index_col=0).dropna(how='any')
        self.df.columns = ['phrase', 'phones', 'file_name']
        self.audio_paths = self._search_files()
        self.sample_rate = sample_rate
        self.max_s_length = max_s_length

    def _search_files(self):
        paths = []

        for fstem in self.df['file_name']:
            path = Path(self.audio_dir) / f'{fstem}.wav'
            paths.append(path)
        return paths

    def build_spectrograms(self, n_files=None, verbose=False):
        inputs = []
        phone_representations = []
        input_lengths = []

        if n_files is None:
            n_files = len(self.audio_paths)

        for a_path in self.audio_paths[:n_files]:
            # Spectrogram: (1, 128, ts)
            fstem, spectrogram, input_length = self._build_spectrogram(a_path, verbose)
            spectrogram = spectrogram.permute(2, 0, 1)
            phone_repr = self.df[self.df['file_name'] == fstem]['phones'].values[0]

            # inputs: (n, 1, nmels, ts)
            # phone_representations: (n, ts)
            inputs.append(spectrogram)
            phone_representations.append(phone_repr)

            input_lengths.append(input_length)
        return inputs, phone_representations, input_lengths

    def _build_spectrogram(self, audio_path, verbose=False):
        # signal: (n_channels, n_samples)
        signal, sr = torchaudio.load(audio_path)
        label = audio_path.stem
        signal = self._resample_if_necessary(signal, sr)
        # signal: (n_channels, n_samples) -> (1, n_samples)
        signal = self._mix_down_if_necessary(signal)
        signal = self.transform(signal) # 1, nmels, ts
        signal_length = signal.size(2)
        # signal = self._pad_if_necessary(signal, label)
        signal = self.amplitude_to_db_transform(signal)
        if verbose:
            print(f'signal: {audio_path}, max: {signal.max()}')
        # signal = signal / torch.max(signal)

        return label, signal, signal_length

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _pad_if_necessary(self, signal, label):
        seq_length = self.max_s_length * self.sample_rate
        padded = torch.zeros((1, seq_length))
        end_idx = signal.size(1)
        if seq_length < end_idx:
            raise Exception(f"File {label} is longer than {self.max_s_length} seconds.")
        padded[0][:end_idx] = signal[0]
        return padded

    def plot_spectrogram(self, spec, title=None, y_label="freq_bin", ax=None):
        # specgram: (1, n_mels, ts)
        spec = spec[0].numpy()
        if ax is None:
            _, ax = plt.subplots(1, 1)
        if title is not None:
            ax.set_title(title)
        ax.set_ylabel(y_label)
        ax.imshow(librosa.power_to_db(spec), origin="lower", aspect="auto", interpolation="nearest")
        plt.show()


if __name__ == "__main__":
    AUDIO_DIR = '../data/audio'
    DATA_PATH = '../data/word_to_phones_dict.csv'
    SAMPLE_RATE = 16000

    mel_transform = MelTransform(AUDIO_DIR, DATA_PATH, sample_rate=SAMPLE_RATE)
    specs, fnames, targets = mel_transform.build_spectrograms(2)
    print(targets)
