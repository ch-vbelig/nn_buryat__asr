from utils.meltransform import MelTransform
from utils.converter import PhoneConverter
from utils import config
from sklearn.utils import shuffle
import numpy as np
import json
import torch
from torch.nn.utils.rnn import pad_sequence

def shuffle_data(data, phones):
    a, b = shuffle(data, phones)
    return a, b

def join_data(arr1, arr2):
    data = []
    for seq1, seq2 in zip(arr1, arr2):
        joined = seq1 + seq2
        data.append(joined)
    return data

def generate_data(inputs, outputs, iter=2):
    x = []
    y = []
    ins = inputs.copy()
    outs = outputs.copy()
    for i in range(iter):
        data, targets = shuffle_data(ins, outs)
        joined_data = join_data(inputs, data)
        joined_targets = join_data(outputs, targets)
        x.extend(joined_data)
        y.extend(joined_targets)

    return x, y

def convert_to_indexes(phones, converter, MAX_SEQUENCE_LENGTH=50):
    targets = []
    target_lengths = []
    for phone_seq in phones:
        ids = np.zeros(MAX_SEQUENCE_LENGTH, dtype=np.int32)
        indexes = [converter.el_to_index[phone] for phone in phone_seq.split()]
        # indexes.append(converter.sil_token)
        ids[:len(indexes)] = indexes

        targets.append(ids.tolist())
        target_lengths.append(len(indexes))
    return targets, target_lengths

def normalize_data(mels):

    data = np.array(mels)

    """
    Max value: 77.63
    Min value: -30.66
    Mean value: -9.52
    Std value: 11.72
    """
    print(f'Max value: {data.max():.2f}')
    print(f'Min value: {data.min():.2f}')
    print(f'Mean value: {data.mean():.2f}')
    print(f'Std value: {data.std():.2f}')

    mean = data.mean()
    std = data.std()

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



if __name__ == '__main__':
    AUDIO_DIR = './data/audio'
    DATA_PATH = 'data/bur_phrase_to_phone.csv'
    PHONE_SET_PATH = './data/bur_phone_set.txt'
    DATA_SAVE_PATH = 'data/data.json'


    SAMPLE_RATE = 16000
    meltransform = MelTransform(
        audio_dir=AUDIO_DIR,
        data_csv=DATA_PATH,
        sample_rate=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS
    )

    # get mel spectrograms
    S_mels, phones, input_lengths = meltransform.build_spectrograms(n_files=None)

    S_mels = torch.nn.utils.rnn.pad_sequence(S_mels)
    S_mels = S_mels.permute(1, 2, 3, 0)
    #
    phone_converter = PhoneConverter(PHONE_SET_PATH)

    # convert to indexes
    targets, target_lengths = convert_to_indexes(phones, phone_converter, MAX_SEQUENCE_LENGTH=60)

    # shuffle
    # data, targets = generate_data(S_mels, targets, 4)

    print(S_mels.size())
    print(torch.tensor(targets).size())

    S_mels = normalize_data(S_mels)

    # save data
    save_data(DATA_SAVE_PATH, S_mels, targets, input_lengths, target_lengths)

    print('class num:', len(phone_converter.el_to_index))
