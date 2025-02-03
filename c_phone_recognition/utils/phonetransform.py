from pathlib import Path
import json

class PhoneTransform:

    def __init__(self, phone_map_path=None):
        self.sil_idx = "0"
        self.space_idx = "1"
        self.phone_to_index = {"<SIL>": self.sil_idx, "<SPACE>": self.space_idx}
        self.index_to_phone = {self.sil_idx: "<SIL>", self.space_idx: "<SPACE>"}
        self.n_elements = len(self.index_to_phone)

        self.phone_map_path = phone_map_path
        self._init_mappings()

    def preprocess(self, fpath):
        ids = []
        phones = self._read_phones_from_file(fpath)
        phone_pairs = self._bigrams(phones)

        for phone in phone_pairs:
            if not phone in self.phone_to_index:
                self._add_element(phone)
            ids.append(self.phone_to_index[phone])

        return ids, self.n_elements

    def _init_mappings(self):
        if self.phone_map_path and Path(self.phone_map_path).exists():
            # load data
            with open(self.phone_map_path) as fp:
                obj = json.load(fp)

            # initialize
            self.phone_to_index = obj['phone_to_index']
            self.index_to_phone = obj['index_to_phone']
            self.n_elements = len(self.index_to_phone)
            print('Loaded mappings')
            print(self.index_to_phone)

    def _read_phones_from_file(self, fpath):
        with open(fpath) as fp:
            text = fp.read().strip()
            phones = text.split()
        return phones

    def _add_element(self, element):
        if element not in self.phone_to_index:
            self.phone_to_index[element] = str(self.n_elements)
            self.index_to_phone[str(self.n_elements)] = element
            self.n_elements += 1

    def _bigrams(self, phones):
        bigrams = []
        for i in range(0, len(phones), 2):
            if i + 1 == len(phones):
                bigrams.append(phones[i])
                break
            elif self.index_to_phone[str(self.space_idx)] in [phones[i], phones[i + 1]]:
                bigrams.extend([phones[i], phones[i + 1]])
            else:
                bigram = ' '.join([phones[i], phones[i + 1]])
                bigrams.append(bigram)
        return bigrams

    def save_maps(self):
        with open(self.phone_map_path, 'w') as fp:
            obj = {
                'phone_to_index': self.phone_to_index,
                'index_to_phone': self.index_to_phone
            }
            json.dump(obj, fp)

    def decode(self, ids):
        _tokens = [self.index_to_phone[str(i)] for i in ids if str(i) in self.index_to_phone]
        tokens = []
        for t in _tokens:
            if len(tokens) > 0 and tokens[-1] == t:
                continue
            tokens.append(t)
        return ' '.join(tokens)


if __name__ == '__main__':
    path = '../data/phones/speaker_id_0_2.txt'
    transform = PhoneTransform()
    transform.preprocess(path)