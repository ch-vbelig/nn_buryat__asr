import json
from b_text_to_phones.utils.converter import Converter

DATA_SAVE_PATH = './data_preprocessed/train_data_adjusted.json'
DATA_LOAD_PATH = './data_raw/bur_word_to_phones_dict.txt'
BUR_ALPHABET_PATH = './data_raw/bur_alphabet.txt'
BUR_PHONE_SET_PATH = './data_raw/bur_phone_set.txt'


def open_file(fpath):
    with open(fpath) as fp:
        text = fp.read()
        return text


def get_pairs(text):
    pairs = []

    for line in text.split('\n'):
        tokens = [p.strip() for p in line.split()]

        if len(tokens) < 2:
            continue

        # get sequence of letters & sequence of phones
        word = tokens[0].lower()
        phones = tokens[1:]

        seq_of_letters = list(word)
        seq_of_phones = phones

        pairs.append([seq_of_letters, seq_of_phones])

    return pairs


def convert_sequence(seq, to_index):
    res = []
    for el in seq:
        idx = to_index[el]
        res.append(idx)
    return res, len(seq)


def convert_pairs(pairs, letter_converter: Converter, phone_converter: Converter):
    res = []
    max_length = 0

    for pair in pairs:
        letter_idxs, letter_seq_length = convert_sequence(pair[0], letter_converter.el_to_index)
        phone_idx, phone_seq_length = convert_sequence(pair[1], phone_converter.el_to_index)

        if max_length < letter_seq_length:
            max_length = letter_seq_length
        elif max_length < phone_seq_length:
            max_length = phone_seq_length

        res.append([letter_idxs, phone_idx])

    return res, max_length


def save_data(fpath, pairs):
    with open(fpath, 'w') as fp:
        data = {
            'data': pairs
        }
        json.dump(data, fp)

if __name__ == '__main__':
    # open file with phone mappings
    text = open_file(DATA_LOAD_PATH)

    # divide into pairs
    pairs = get_pairs(text)
    print(pairs)

    # get converters
    letter_converter = Converter(BUR_ALPHABET_PATH)
    phone_converter = Converter(BUR_PHONE_SET_PATH)

    # convert pairs into indexes
    pairs, max_length = convert_pairs(pairs, letter_converter, phone_converter)
    print(pairs[:1])
    print(max_length)

    # save data
    save_data(DATA_SAVE_PATH, pairs)
