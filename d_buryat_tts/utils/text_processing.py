"""
Utility for text normalization.
"""

import re
from pathlib import Path
import sys
import argparse
from d_buryat_tts.utils.config import Config

def vocab_lookup(text):
    vocab_char_to_idx = {char: idx for idx, char in enumerate(Config.vocab)}
    return [vocab_char_to_idx[char] for char in text]

def read_file(fpath):
    with open(fpath, encoding='utf-8') as fp:
        text = fp.read().strip()
        return text

def save_file(fpath, content):
    with open(fpath, 'w', encoding='utf-8') as fp:
        fp.write(content)

def normalize(text):
    text = text.lower()
    # substitute characters 'h' and 'y' with 'һ' and 'ү'
    text = re.sub('h', 'һ', text)
    text = re.sub('y', 'ү', text)

    # remove numbers english characters
    text = re.sub(r'[asdf-z0-9_]', '', text)

    # remove non-alphabetic characters
    text = re.sub(r'[^\w ]', '', text)

    # remove double spaces
    text = re.sub(r'\s+\s+', ' ', text)

    text = text.strip()
    return text


def load_lexicon(lexicon_file):
    with open(lexicon_file, encoding='utf-8') as fp:
        lines = fp.readlines()
        lines = sorted([(line.split()[0], ' '.join(line.split()[1:]).strip()) for line in lines])
        lex = dict(lines)
        return lex

def get_phones(text, lexicon):
    normalized = re.sub(r'[.…,!?;:()«»“”"–—\-•_/`~@#$%^&*\'\\]', ' ', text)
    words = re.split(r'\W', normalized)
    words = [word for word in words if len(word) > 0]
    phones = [lexicon[word] for word in words if word in lexicon]
    unknown_words = [word for word in words if not word in lexicon]

    return ' | '.join(phones), unknown_words


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performs text preprocess from raw .txt files')
    parser.add_argument('-l', '--load',
                        dest='load_path',
                        required=False,
                        default='data/raw_text',
                        help='Directory to raw text files')
    parser.add_argument('-s', '--save',
                        dest='save_path',
                        required=False,
                        default='data/text',
                        help='Directory to save text files')
    args = parser.parse_args()

    # Get .txt files
    bur_files = list(Path(args.load_path).glob('*.txt'))
    text_lengths = []
    for fpath in bur_files:
        print("Processing:", fpath.name)
        text = read_file(fpath)
        text = normalize(text)

        text_lengths.append(len(text))

        if not set(text) <= set(Config.vocab):
            raise RuntimeError(f"{fpath.name} contains non-vocab characters")
        else:
            save_path = Path(args.save_path) / fpath.name
            save_file(save_path, text)

    print(max(text_lengths))

#
# if __name__ == "__main__":
#     import os
#     with open(sys.argv[1], "r") as f:
#         text = f.read()
#     lines = split_text(text, max_len=int(sys.argv[2]))
#     path = os.path.join(os.path.dirname(sys.argv[1]), "lines.txt")
#     with open(path, "w") as f:
#         for line in lines:
#             f.write(line + "\n")