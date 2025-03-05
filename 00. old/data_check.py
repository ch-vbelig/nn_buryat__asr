import re
import pandas as pd


def open_bur_phone_set(path):
    with open(path) as fp:
        lines = fp.readlines()
        phones = [line.strip() for line in lines]
        phones.append('<SPACE>')
        phones = set(phones)
    return phones


def open_csv(path):
    df = pd.read_csv(path, header=None, index_col=0)
    df.columns = ['speaker_id', 'phrase', 'phones', 'file_name']
    return df


def check():
    DATA_PATH = 'bur_phrase_to_phone.csv'
    BUR_PHONE_SET_PATH = '../c_phone_recognition/data/bur_phone_set.txt'

    df = open_csv(DATA_PATH)
    phone_set = open_bur_phone_set(BUR_PHONE_SET_PATH)

    for i, phone_row in enumerate(df['phones']):
        phones = set(phone_row.split())

        if not phones.issubset(phone_set):
            print(df.iloc[i]['file_name'], ':', phones - phone_set)

if __name__ == '__main__':
    check()