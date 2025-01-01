from b_text_to_phones.utils.converter import Converter
from b_text_to_phones import bpreprocess as preprocess

DATA_LOAD_PATH = 'data_raw/bur_word_to_phones_dict.txt'
BUR_ALPHABET_PATH = 'data_raw/bur_alphabet.txt'
BUR_PHONE_SET_PATH = 'data_raw/bur_phone_set.txt'


def filter_pairs(pairs, letter_converter: Converter, phone_converter: Converter):
    thrash_count = 0
    pairs_filtered = []

    for i, (inp, tgt) in enumerate(pairs):
        set_inp = set(list(inp))
        set_letter = set(list(letter_converter.el_to_index.keys()))

        set_tgt = set(list(tgt))
        set_phone = set(list(phone_converter.el_to_index.keys()))

        if not (set_inp <= set_letter and set_tgt <= set_phone):
            thrash_count += 1
            print(i, inp, tgt)
            continue

        pairs_filtered.append([inp, tgt])

    return pairs_filtered, thrash_count


def save_pairs(pairs, fpath):
    text = ""

    for (inp, tgt) in pairs:
        text += ''.join(inp)
        text += ' '
        text += ' '.join(tgt)
        text += '\n'

    with open(fpath, 'w') as fp:
        fp.write(text)


if __name__ == '__main__':
    # open file with phone mappings
    text = preprocess.open_file(DATA_LOAD_PATH)

    # divide into pairs
    pairs = preprocess.get_pairs(text)

    # get converters
    letter_converter = Converter(BUR_ALPHABET_PATH)
    phone_converter = Converter(BUR_PHONE_SET_PATH)

    pairs_filtered, t_count = filter_pairs(pairs, letter_converter, phone_converter)
    print(len(pairs_filtered))
    print("Thrash: ", t_count)

    # save_pairs(pairs_filtered, DATA_LOAD_PATH)
