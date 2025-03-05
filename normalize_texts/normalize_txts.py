import re
from pathlib import Path



def read_file(fpath):
    with open(fpath, encoding='utf-8') as fp:
        text = fp.read().strip()
        return text

def save_file(fpath, content):
    with open(fpath, 'w', encoding='utf-8') as fp:
        fp.write(content)

def normalize(text):
    text = text.lower()
    text = re.sub('h', 'һ', text)
    text = re.sub('y', 'ү', text)
    text = re.sub(r'\s+\s+', ' ', text)
    return text


def load_lexicon(lexicon_file):
    with open(lexicon_file, encoding='utf-8') as fp:
        lines = fp.readlines()
        lines = sorted([(line.split()[0], ' '.join(line.split()[1:]).strip()) for line in lines])
        lex = dict(lines)
        return lex

def get_phones(text, lexicon):
    normalized = re.sub(r'[.…,!?;:()«»“”"–—\-•/`~@#$%^&*\'\\]', ' ', text)
    words = re.split(r'\W', normalized)
    words = [word for word in words if len(word) > 0]
    phones = [lexicon[word] for word in words if word in lexicon]
    unknown_words = [word for word in words if not word in lexicon]

    return ' | '.join(phones), unknown_words


if __name__ == '__main__':
    TEXT_DIR = './texts'
    PHONE_DIR = './phones'
    LEXICON_FILE = 'bur_lexicon.txt'

    # Flags
    text_normalization = True
    lexicon_added = True

    # Load lexicon
    lexicon = load_lexicon(LEXICON_FILE)

    # Get .txt files
    bur_files = list(Path(TEXT_DIR).glob('*.txt'))

    unknown_words = []

    for fpath in bur_files:
        text = read_file(fpath)
        text = normalize(text)

        if text_normalization:
            save_file(fpath, text)

        if not lexicon_added:
            _, unknown = get_phones(text, lexicon)
            unknown_words.extend(unknown)
        else:
            phones, _ = get_phones(text, lexicon)
            phone_path = Path(PHONE_DIR) / fpath.name
            save_file(phone_path, phones)

    if len(unknown_words) > 0:
        unknown_words = list(set(unknown_words))
        print('\n'.join(unknown_words))