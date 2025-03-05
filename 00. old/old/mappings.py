import json
import itertools


def get_phones(path):
    with open(path) as fp:
        text = fp.read().strip()
        phones = text.split()
    return phones


def get_phone_pairs(phones):
    phone_to_index = {
        "<SIL>": 0,
        "<SPACE>": 1
    }
    n_classes = 2

    mix = list(set(itertools.product(phones, phones)))
    mix = [f'{m[0]} {m[1]}' for m in mix]

    for phone in phones:
        if not phone in phone_to_index:
            phone_to_index[phone] = n_classes
            n_classes += 1

    for bigram in mix:
        if not bigram in phone_to_index:
            phone_to_index[bigram] = n_classes
            n_classes += 1

    return phone_to_index

def get_phone_dict(phones):
    phone_to_index = {
        "<SIL>": 0,
        "<SPACE>": 1
    }
    n_classes = 2

    for phone in phones:
        if not phone in phone_to_index:
            phone_to_index[phone] = n_classes
            n_classes += 1

    return phone_to_index


def save_dictionary(path, phone_to_index, index_to_phone):
    obj = {}
    obj['phone_to_index'] = phone_to_index
    obj['index_to_phone'] = index_to_phone

    with open(path, 'w') as fp:
        json.dump(obj, fp)




if __name__ == '__main__':
    PATH = '../../c_phone_recognition/data/bur_phone_set.txt'
    MAP_SAVE_PATH = '../../c_phone_recognition/data/phones_map.json'

    phones = get_phones(PATH)

    # phone_to_index = get_phone_pairs(phones)
    phone_to_index = get_phone_dict(phones)
    index_to_phone = {i: p for p, i in phone_to_index.items()}
    print(len(phone_to_index))
    print(index_to_phone)

    save_dictionary(MAP_SAVE_PATH, phone_to_index, index_to_phone)

