from utils.meltransform import MelTransform
from utils.converter import PhoneConverter
from utils import config
import numpy as np
import json
import torch


def convert_to_indexes(phones, converter, max_seq_length=config.MAX_TARGET_LENGTH):
    """
    :param: phones (list): list of phone representations for all audio files in audio directory
    :param: converter (PhoneConverter): to/from phone converter
    :param: max_seq_length (int): the length of padded sequence
    :return: targets (list): list of converted phone representations for all files in audio directory
    :return: target_lengths (list of int): length of target sequences before padding
    """

    targets = []
    target_lengths = []

    for phone_seq in phones:
        # convert and pad with zeros
        ids = np.zeros(max_seq_length, dtype=np.int32)
        indexes = [converter.phone_to_index[phone] for phone in phone_seq.split()]
        indexes.append(converter.sil_idx)
        ids[:len(indexes)] = indexes

        targets.append(ids.tolist())
        target_lengths.append(len(indexes))
    return targets, target_lengths


def normalize_data(mels, verbose=False):
    data = np.array(mels)

    max_val = data.max()
    min_val = data.min()
    mean = data.mean()
    std = data.std()

    if verbose:
        print(f"Max value: {max_val}.")
        print(f"Min value: {min_val}.")
        print(f"Mean value: {mean}.")
        print(f"Std value: {std}.")

    data = (data - mean) / std

    return data.tolist()


def save_data(fpath, data, targets, input_lengths, target_lengths):
    obj = {
        "data": data,
        "targets": targets,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths
    }
    with open(fpath, 'w') as fp:
        json.dump(obj, fp)


def run_preprocess():
    AUDIO_DIR = './data/audio'
    DATA_PATH = 'data/bur_phrase_to_phone.csv'
    PHONE_SET_PATH = './data/bur_phone_set.txt'
    DATA_SAVE_PATH = 'data/data_2.json'

    SAMPLE_RATE = 16000
    meltransform = MelTransform(
        audio_dir=AUDIO_DIR,
        data_csv=DATA_PATH,
        sample_rate=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS
    )

    # Calculate melspectrograms
    # S_mels (list of tensors): (n, ) (ts, 1, n_mels)
    S_mels, phones, input_lengths = meltransform.build_spectrograms(n_files=None)

    S_mels = torch.nn.utils.rnn.pad_sequence(S_mels)
    S_mels = S_mels.permute(1, 2, 3, 0)  # S_mels (tensor): (n, 1, n_mels, ts)

    phone_converter = PhoneConverter(PHONE_SET_PATH)

    # Convert phones into indexes
    targets, target_lengths = convert_to_indexes(phones, phone_converter, max_seq_length=config.MAX_TARGET_LENGTH)

    print(S_mels.size())
    print(torch.tensor(targets).size())
    print('class num:', len(phone_converter.phone_to_index))

    # Perform normalization
    S_mels = normalize_data(S_mels, verbose=False)

    # save data
    save_data(DATA_SAVE_PATH, S_mels, targets, input_lengths, target_lengths)


if __name__ == '__main__':
    run_preprocess()

