import torch
import torchaudio
import json
from pathlib import Path
import pandas as pd
import numpy as np
import utils.config as config
from utils.converter import PhoneConverter

PHONE_SET_PATH = './data/bur_phone_set.txt'


class SpeechDataset:
    def __init__(self, data_path):
        """
        :param data_path: path to *.json file with preprocessed data
        :key 'data': melspectrograms : (n, 1, n_mels, ts), where 1 is number of channels
        :key 'targets': indexes of target phones (padded): (n, max_seq_length)
        :key 'input_length': length of each input sequence (when non-padded)
        :key 'target_length': length of each output sequence (when non-padded)
        """
        self.data_path = data_path
        obj = self.load_data()

        self.data = obj['data']
        self.targets = obj['targets']
        self.input_lengths = obj['input_lengths']
        self.target_lengths = obj['target_lengths']

    def load_data(self):
        with open(self.data_path) as fp:
            obj = json.load(fp)
            return obj

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        x = torch.tensor(self.data[item], dtype=torch.float)
        y = torch.tensor(self.targets[item], dtype=torch.long)
        input_lengths = torch.tensor(self.input_lengths[item], dtype=torch.long)
        target_lengths = torch.tensor(self.target_lengths[item], dtype=torch.long)
        return x, y, input_lengths, target_lengths


class SpeechDatasetWav2Vec:
    def __init__(self, audio_dir, data_csv, ):
        """
        :param audio_dir: path to *.json file with preprocessed data
        :key 'data': melspectrograms : (n, 1, n_mels, ts), where 1 is number of channels
        :key 'targets': indexes of target phones (padded): (n, max_seq_length)
        :key 'input_length': length of each input sequence (when non-padded)
        :key 'target_length': length of each output sequence (when non-padded)
        """
        self.audio_dir = audio_dir

        self.data_csv_path = data_csv
        self.df = pd.read_csv(data_csv, header=None, index_col=0).dropna(how='any')
        self.df.columns = ['speaker_id', 'phrase', 'phones', 'file_name']
        self.audio_paths = self._get_audio_paths()

    def _get_audio_paths(self):
        path = Path(self.audio_dir)
        audio_paths = list(path.glob('*.wav'))
        audio_paths = [path for path in audio_paths if (path.stem in self.df['file_name'].tolist())]
        print(audio_paths)
        return audio_paths

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, item):
        waveform, sr = torchaudio.load(self.audio_paths[item])
        waveform = self._resample_if_necessary(waveform, sr)
        waveform = self._mix_down_if_necessary(waveform)  # waveform: (n_channels, n_samples) : (1, n_samples)
        waveform = self._pad_if_necessary(waveform)  # waveform: (n_channels, n_samples) : (1, n_samples)

        file_stem = self.audio_paths[item].stem
        phone_repr = self.df[self.df['file_name'] == file_stem]['phones'].values[0]

        phone_converter = PhoneConverter(PHONE_SET_PATH)
        target, target_length = convert_to_indexes(phone_repr, phone_converter,
                                                     max_seq_length=config.MAX_TARGET_LENGTH)

        x = waveform.squeeze()
        y = torch.tensor(target, dtype=torch.long)
        input_lengths = torch.tensor(waveform.size(1), dtype=torch.long)
        target_lengths = torch.tensor(target_length, dtype=torch.long)
        return x, y, input_lengths, target_lengths

    def _resample_if_necessary(self, signal, sr, target_sr=16000):
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _pad_if_necessary(self, signal):
        max_length = config.SAMPLE_RATE * 10
        padded = torch.zeros((signal.size(0), max_length))

        end = signal.size(1) if max_length > signal.size(1) else max_length

        padded[0][:end] = signal[0][:end]
        return padded


def convert_to_indexes(phones, converter, max_seq_length=config.MAX_TARGET_LENGTH):
    """
    :param: phones (list): list of phone representations for all audio files in audio directory
    :param: converter (PhoneConverter): to/from phone converter
    :param: max_seq_length (int): the length of padded sequence
    :return: targets (list): list of converted phone representations for all files in audio directory
    :return: target_lengths (list of int): length of target sequences before padding
    """

    ids = np.zeros(max_seq_length, dtype=np.int32)
    indexes = [converter.phone_to_index[phone] for phone in phones.split()]
    indexes.append(converter.sil_idx)
    ids[:len(indexes)] = indexes

    return ids.tolist(), len(indexes)
