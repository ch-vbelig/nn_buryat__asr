


LEXICON_FILE = f'data/bur_lexicon.txt'
LEXICON_FILE_2 = f'data/phone_lexicon_2.txt'

with open(LEXICON_FILE, encoding='UTF-8') as fp:
    text = fp.read()
    lines = [line.strip() for line in text.split('\n')]
    words = [line.split()[0] for line in lines]

print(words[2805:])

with open(LEXICON_FILE_2, encoding='UTF-8') as fp:
    text = fp.read()
    lines = [line.strip() for line in text.split('\n')]
    phones = [' '.join(line.split()[1:]) for line in lines]

print(phones[2805:])

LEXICON_FILE = f'data/bur_lexicon_2.txt'

pairs = zip(words[2805:], phones[2805:])
pairs = [f'{pair[0]}\t{pair[1]}' for pair in pairs]

text = '\n'.join(pairs)
print(text)

with open(LEXICON_FILE, 'asdf', encoding='UTF-8') as fp:
    fp.write(text)
