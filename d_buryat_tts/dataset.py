import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from collections import OrderedDict
from random import shuffle
from utils.config import Config, VocoderConfig
from d_buryat_tts.utils import text_processing
import random
import librosa
from librosa.util import normalize

class TTSDataset(Dataset):
    """
    Text to speech dataset.

    Args:
        text_dir: Path to the text file containing the lines.
        mel_dir: Directory with all the mel spectrograms.
        lin_dir: Directory with all the linear spectrograms. Set 'None' for the text2mel training, because only mel
            spectrograms are needed there.
    """
    def __init__(self, text_dir, mel_dir, lin_dir=None):
        self.text_dir = text_dir
        self.mel_dir = mel_dir
        self.lin_dir = lin_dir

        self.text_files, self.mel_files, self.lin_files = self._search_files()

    def __len__(self):
        return len(self.text_files)

    def __getitem__(self, idx):
        text_file_stem = self.text_files[idx].stem
        mel_file_stem = self.mel_files[idx].stem

        if text_file_stem != mel_file_stem:
            raise RuntimeError(f"Error: file names do not match: {text_file_stem} VS {mel_file_stem}")

        # Text as tensor: (ts, )
        text_tensor = self._process_text(self.text_files[idx])

        # Mel as tensor: ((ts + num_padding) / 4, 80) => (ts, 80)
        mel_tensor, ts = self._process_mel(self.mel_files[idx])


        data = {
            "text": text_tensor,    # (ts, )
            "mel": mel_tensor   # (ts, 80)
        }

        if self.lin_dir is not None:
            lin_tensor = self._process_lin(self.lin_files[idx], ts)
            data["lin"] = lin_tensor    # (ts, 1025)

        return data

    def _search_files(self):
        text_files = list(Path(self.text_dir).glob('*.txt'))
        mel_files = list(Path(self.mel_dir).glob('*.npy'))

        if self.lin_dir:
            lin_files = list(Path(self.lin_dir).glob('*.npy'))
        else:
            lin_files = None
        return text_files, mel_files, lin_files

    @staticmethod
    def _process_text(text_path):
        with open(text_path, encoding='utf-8') as fp:
            text = fp.read().strip()    # str

        text = text_processing.normalize(text)  # str
        text += Config.vocab_end_of_text    # str

        text_ids = text_processing.vocab_lookup(text)   # ids: (ts, )
        text_ids = torch.tensor(text_ids, dtype=torch.long) # tensor: (ts, )
        return text_ids

    @staticmethod
    def _process_mel(mel_path):
        mel = np.load(mel_path) # (ts, n_mels) : (ts, 80)
        mel = torch.tensor(mel) # (ts, n_mels) : (ts, 80)

        ts = mel.size(0)

        # Marginal padding for reduction
        num_paddings = Config.time_reduction - (ts % Config.time_reduction) \
            if ts % Config.time_reduction != 0 \
            else 0

        # Padded along ts (right padded)
        # mel : (ts, n_mels) : (ts + num_padding, 80)
        mel = F.pad(
            input=mel,
            pad=[0, 0, 0, num_paddings],
            mode='constant',
            value=0
        )

        # Time reduction from T to T // 4
        mel = mel[::Config.time_reduction] # (ts / 4, n_mels)
        return mel, ts

    @staticmethod
    def _process_lin(lin_path, ts):
        lin = np.load(lin_path) # (ts, freq_bins) : (ts, 1025)  # freq_bins = n_fft // 2 + 1
        lin = torch.tensor(lin) #   (ts, 1025)

        # Marginal padding for reduction shape
        num_paddings = Config.time_reduction - (ts % Config.time_reduction) \
            if ts % Config.time_reduction != 0 \
            else 0

        # (ts + num_padding, 1025)
        lin = F.pad(
            input=lin,
            pad=[0, 0, 0, num_paddings],
            mode='constant',
            value=0
        )

        return lin  # (ts, 1025)


def collate_fn(data):
    """
    Creates mini-batch tensors from the list maps.
    Args:
        data: List of string maps containing the training data ("text", "mel", "lin").
    """
    texts = [d["text"] for d in data]   # bs of (ts, )
    mels = [d["mel"] for d in data] # bs of (ts, n_mels)

    # Get max text and mel lengths in batch
    max_text_len = max(text.size(0) for text in texts)
    max_mel_len = max(mel.size(0) for mel in mels)

    # Prepare zero padding for texts, mels and lins
    text_padded = torch.zeros(len(texts), max_text_len, dtype=torch.long)   # (bs, max_text_len)
    mel_padded = torch.zeros(len(mels), max_mel_len, mels[0].size(-1))    # (bs, max_mel_len, n_mels)

    # Pad data
    for i in range(len(texts)):
        text_padded[i, :texts[i].size(0)] = texts[i]
        mel_padded[i, :mels[i].size(0)] = mels[i]


    obj = {
        "text": text_padded, # (bs, max_text_len)
        "mel": mel_padded,   # (bs, max_mel_len, n_mels)
    }

    if "lin" in data[0]:
        lins = [d["lin"] for d in data] # bs of (ts, freq_bins)
        max_lin_len = max(lin.size(0) for lin in lins)

        lin_padded = torch.zeros(len(lins), max_lin_len, lins[0].size(-1))   # (bs, max_lin_len, freq_bins)
        for i in range(len(texts)):
            lin_padded[i, :lins[i].size(0)] = lins[i]

        obj["lin"] = lin_padded    # (bs, max_lin_len, freq_bins) : (bs, max_lin_len, 1025)

    return obj


class BucketBatchSampler(torch.utils.data.Sampler):
    """
    Groups inputs into buckets of equal length and samples batches out of these buckets. This way all inputs in asdf batch
    will have the same size and no padding is needed.
    Adapted from https://discuss.pytorch.org/t/tensorflow-esque-bucket-by-sequence-length/41284

    Args:
        inputs: A 1d array containing the feature that should be used for bucketing (indices in the same order as in the
            data set). In our case we want to bucket text sizes, so 'inputs' will be the list of all texts.

        batch_size: Maximum batch size (note that some batches can be smaller if buckets are not large enough).

        bucket_boundaries: int list, increasing non-negative numbers. The edges of the buckets to use when bucketing
            tensors.  Two extra buckets are created, one for `input_length < bucket_boundaries[0]` and one for
            `input_length >= bucket_boundaries[-1]`.
    """
    def __init__(self, inputs, batch_size, bucket_boundaries):
        self.batch_size = batch_size
        # Add bucket for smaller and larger inputs
        self.bucket_boundaries = [-1] + bucket_boundaries + []
        ind_n_len = []
        for i, p in enumerate(inputs):
            ind_n_len.append((i, p.size(0)))
        self.ind_n_len = ind_n_len
        self.batch_list = self._generate_batch_map()
        self.num_batches = len(self.batch_list)

    def _generate_batch_map(self):
        # Shuffle all of the indices first so they are put into buckets differently
        shuffle(self.ind_n_len)
        # Organize lengths, e.g., batch_map[10] = [30, 124, 203, ...] <= indices of sequences of length 10
        batch_map = OrderedDict()
        for idx, length in self.ind_n_len:
            # Find corresponding bucket
            if length < self.bucket_boundaries[0]:
                bucket = -1
            elif length > self.bucket_boundaries[-1]:
                bucket = self.bucket_boundaries[-1] + 1
            else:
                for i in range(len(self.bucket_boundaries)):
                    if length == self.bucket_boundaries[i]:
                        bucket = i
                        break
                    if length < self.bucket_boundaries[i]:
                        bucket = i - 1
                        break
            # Save index in bucket
            if bucket not in batch_map:
                batch_map[bucket] = [idx]
            else:
                batch_map[bucket].append(idx)
        # Use batch_map to split indices into batches of equal size
        # e.g., for batch_size=3, batch_list = [[23,45,47], [49,50,62], [63,65,66], ...]
        batch_list = []
        for length, indices in batch_map.items():
            for group in [indices[i:(i + self.batch_size)] for i in range(0, len(indices), self.batch_size)]:
                batch_list.append(group)
        return batch_list

    def batch_count(self):
        return self.num_batches

    def __len__(self):
        return len(self.ind_n_len)

    def __iter__(self):
        self.batch_list = self._generate_batch_map()
        # shuffle all the batches so they arent ordered by bucket size
        shuffle(self.batch_list)
        for i in self.batch_list:
            yield i


class _RepeatSampler(object):
    """ Sampler that repeats forever. """
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class FastDataLoader(torch.utils.data.dataloader.DataLoader):
    """
    DataLoader that pretends, that the epoch never ends. This improves performance on windows, because the process
    spawning at the start of each epoch will be avoided. See https://github.com/pytorch/pytorch/issues/15849
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)





class WaveRNNDataset(Dataset):
    def __init__(self, mel_dir, qwav_dir, max_items=None):
        self.mel_dir = mel_dir
        self.qwav_dir = qwav_dir
        self.max_items = max_items
        self.sample_frames = VocoderConfig.sample_frames
        self.hop_length = Config.hop_length

        self.mel_files, self.qwav_files = self._search_files()


    def __len__(self):
        return len(self.mel_files)

    def __getitem__(self, item):
        idx = random.randint(0, len(self.mel_files) - 1)
        mel_path_stem = self.mel_files[idx].stem
        qwav_path_stem = self.qwav_files[idx].stem

        if mel_path_stem != qwav_path_stem:
            raise RuntimeError(f"Error: file names do not match: {mel_path_stem} VS {qwav_path_stem}")

        mel = np.load(self.mel_files[idx]) # (ts, 80)
        qwav = np.load(self.qwav_files[idx]) # (n_samples, )

        pos = random.randint(0, mel.shape[0] - self.sample_frames - 1)
        start, end = pos, pos + self.sample_frames

        mel = mel[start: end, :] # (ts, 80)
        qwav = qwav[start * self.hop_length: end * self.hop_length + 1]


        # mel: (ts, 80)
        # qwav: (n_samples)
        return mel, qwav



    def _search_files(self):
        mel_files = list(Path(self.mel_dir).glob('*.npy'))
        qwav_files = list(Path(self.qwav_dir).glob('*.npy'))

        if self.max_items is not None:
            assert self.max_items < len(mel_files), f'Max items exceeds actual number of files!'

            ids = random.sample(range(len(mel_files)), k=self.max_items)
            mel_files = [mel_files[i] for i, _ in enumerate(mel_files) if i in ids]
            qwav_files = [qwav_files[i] for i, _ in enumerate(qwav_files) if i in ids]
        return mel_files, qwav_files



class MelGANDataset(Dataset):
    def __init__(self, wav_dir, segment_length, augment=True):
        self.wav_dir = wav_dir
        # self.mel_dir = mel_dir
        self.wav_files = self._search_files()
        self.segment_length = segment_length
        self.augment = augment

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        wav = self._load_wav(self.wav_files[idx]) # (n_samples, )
        wav_length = wav.size(0)

        if wav.size(0) >= self.segment_length:
            start = random.randint(0, wav_length - self.segment_length)
            wav = wav[start : start + self.segment_length]
        else:
            wav = F.pad(wav, (0, self.segment_length - wav_length), "constant").detach()

        wav = wav.unsqueeze(0) # (1, n_samples)

        return wav

    def _load_wav(self, path):
        wav, _ = librosa.load(path, sr=Config.sample_rate)  # (n_samples, )
        wav = 0.95 * normalize(wav)

        if self.augment:
            amplitude = np.random.uniform(low=0.3, high=1.0)
            wav = wav * amplitude

        return torch.from_numpy(wav).float()

    def _search_files(self):
        wav_files = list(Path(self.wav_dir).glob('*.wav'))
        # mel_files = list(Path(self.mel_dir).glob('*.npy'))
        return wav_files


if __name__ == '__main__':
    dataset = MelGANDataset(
        wav_dir='C:\\Users\\buryat_saram\\Music\\Project Buryat Saram\\buryat_text_to_speech\\raw_data_buryat_wavs',
        segment_length=32
    )

    m = dataset[0]

    print(m.size())