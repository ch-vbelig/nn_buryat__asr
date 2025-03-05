import re

lexicon_path = 'data/bur_lexicon.txt'
phone_lexicon_path = 'data/phone_lexicon.txt'

with open(lexicon_path, encoding='UTF-8') as fp:
    text = fp.read()

text = re.sub(r'[sh]', '', text)
text = re.sub(r'(A|O)E', r'\1E', text)
text = re.sub(r'UI', r'U I', text)

with open(lexicon_path, 'w', encoding='UTF-8') as fp:
    fp.write(text)

lines = text.split('\n')

phones = ['_'.join(line.split()[1:]) for line in lines]
print(phones)

text = '\n'.join(phones)
with open(phone_lexicon_path, 'w') as fp:
    fp.write(text)
