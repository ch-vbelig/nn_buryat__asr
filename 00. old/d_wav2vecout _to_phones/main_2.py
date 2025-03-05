import re

lexicon_path = 'data/phone_lexicon.txt'
phone_lexicon_path = 'data/phone_lexicon_2.txt'

with open(lexicon_path, encoding='UTF-8') as fp:
    text = fp.read()
    phones = text.split('\n')

acoustics = []

for phone in phones:
    temp = re.sub('_', ' ', phone)
    temp = re.sub(r' H ', ' C ', temp)
    temp = re.sub(r'HH', 'H', temp)
    temp = re.sub(r'OE', 'O E', temp)
    temp = re.sub(r'ZH', 'Z H', temp)
    # temp = re.sub('Z', 'S', temp)
    temp = re.sub('EH', 'E', temp)
    temp = re.sub(r'([A-Z])J', r'\1', temp)
    temp = re.sub('Y', 'I', temp)
    temp = re.sub('J', 'Y', temp)
    temp = re.sub('Gr', 'G', temp)
    temp = re.sub('NG', 'N G', temp)
    temp = re.sub('CH', 'C H', temp)
    temp = re.sub('SCH', 'S H', temp)
    temp = re.sub('SH', 'S H', temp)
    temp = re.sub('SC', 'S', temp)
    temp = re.sub(r'(A|U|E|O|I){2}', r'\1', temp)
    temp = re.sub(r'([A-Z])J', r'\1', temp)
    acoustics.append(temp)



final = []

for phone, acoustic in zip(phones, acoustics):
    final.append(f"{phone}\t{acoustic} |")


final = '\n'.join(final)

with open(phone_lexicon_path, 'w') as fp:
    fp.write(final)
