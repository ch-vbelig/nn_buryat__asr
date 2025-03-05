import torch
from pathlib import Path
from utils.phonetransform import PhoneTransform
from utils.meltransform import MelTransform
import utils.config as config
from torch.utils.data import Dataset
from sklearn.utils import shuffle

class SpeechDataset(Dataset):
    def __init__(self, audio_dir, phone_dir, phone_map_path):
        """
        :param self.audio_dir: path to audio directory
        :param self.phone_dir: path to texts directory
        :param self.phone_map_path: path to save/load existing maps
        :param self.meltransform (MelTransform): meltransform
        :param self.phonetransform (PhoneTransform): phonetransform
        """
        self.audio_dir = audio_dir
        self.phone_dir = phone_dir

        self.audio_files, self.phone_files = self._search_files()

        self.meltransform = MelTransform()
        self.phonetransform = PhoneTransform(phone_map_path)

        self.n_classes = 0

    def save_maps(self):
        self.phonetransform.save_maps()

    def _search_files(self):
        audio_files = list(Path(self.audio_dir).glob('*.wav'))
        phone_files = list(Path(self.phone_dir).glob('*.txt'))

        # audio_files = audio_files * 3
        # phone_files = phone_files * 3

        audio_files, phone_files = shuffle(audio_files, phone_files)

        return audio_files, phone_files

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, item):
        audio_path = self.audio_files[item]
        phone_path = self.phone_files[item]

        # x: (ts, n_channels, n_mels)
        x = self.meltransform.build_spectrogram(audio_path, time_first=True)
        x = (x - config.MEAN_VALUE) / config.STD_VALUE

        # y: list of ids -> tensor of ids (ts)
        y, n_classes = self.phonetransform.preprocess(phone_path, bigram=False)
        y = torch.tensor(y, dtype=torch.long)

        # get number of classes (bigrams)
        if self.n_classes < n_classes:
            self.n_classes = n_classes

        input_length = x.size(0)
        target_length = y.size(0)

        return x, y, input_length, target_length














class Wav2VecDataset(Dataset):
    def __init__(self, audio_dir, phone_dir, phone_map_path):
        """
        :param self.audio_dir: path to audio directory
        :param self.phone_dir: path to texts directory
        :param self.phone_map_path: path to save/load existing maps
        :param self.meltransform (MelTransform): meltransform
        :param self.phonetransform (PhoneTransform): phonetransform
        """
        self.audio_dir = audio_dir
        self.phone_dir = phone_dir

        self.audio_files, self.phone_files = self._search_files()

        self.meltransform = MelTransform()
        self.phonetransform = PhoneTransform(phone_map_path)

        self.n_classes = 0

    def save_maps(self):
        self.phonetransform.save_maps()

    def _search_files(self):
        audio_files = tuple(Path(self.audio_dir).glob('*.wav'))
        phone_files = tuple(Path(self.phone_dir).glob('*.txt'))

        return audio_files, phone_files

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, item):
        audio_path = self.audio_files[item]
        phone_path = self.phone_files[item]

        # x: (ts,)
        x = self.meltransform.build_waveform(audio_path, time_first=True)

        # y: list of ids -> tensor of ids (ts)
        y, n_classes = self.phonetransform.preprocess(phone_path, bigram=False)
        y = torch.tensor(y, dtype=torch.long)

        # get number of classes (bigrams)
        if self.n_classes < n_classes:
            self.n_classes = n_classes

        input_length = x.size(0)
        target_length = y.size(0)

        return x, y, input_length, target_length