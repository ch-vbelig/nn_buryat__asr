import b_text_to_phones.dengine as engine
from b_text_to_phones.utils.converter import Converter
from b_text_to_phones.cmodel import EncoderRNN, AttnDecoderRNN
import b_text_to_phones.utils.config as config
import re

BUR_ALPHABET_PATH = '../data_raw/bur_alphabet.txt'
BUR_PHONE_SET_PATH = '../data_raw/bur_phone_set.txt'
DATA_PATH = '../data_raw/predict_phones.txt'
DATA_SAVE_PATH = '../data_raw/predicted_phones.txt'


def open_file(fpath):
    with open(fpath) as fp:
        return fp.read()

def save_to_file(fpath, words, predictions):
    with open(fpath, 'w') as fp:
        for word, phone in zip(words, predictions):
            phone_line = ' '.join(phone[:-1])
            line = f'{word}\t{phone_line}\n'
            print(line)
            fp.write(line)

def load_models():
    letter_converter = Converter(BUR_ALPHABET_PATH)
    phone_converter = Converter(BUR_PHONE_SET_PATH)

    encoder = EncoderRNN(
        input_vocab_size=letter_converter.n_elements,
        hidden_size=config.HIDDEN_SIZE
    )

    decoder = AttnDecoderRNN(
        output_vocab_size=phone_converter.n_elements,
        hidden_size=config.HIDDEN_SIZE
    )

    encoder = engine.load_model(encoder, config.ECODER_SAVE_PATH)
    decoder = engine.load_model(decoder, config.ATTN_DECODER_SAVE_PATH)

    return encoder, decoder, letter_converter, phone_converter

def get_words(text):
    words = [w.strip() for w in text.split('\n')]
    words = [w.lower() for w in words]
    words = [w for w in words if re.match('\w+', w)]
    words = [w for w in words if re.match('\D+', w)]
    return words

def get_predictions(words, encoder, decoder, letter_converter, phone_converter):
    predictions = []

    for word in words:
        output = engine.predict(encoder, decoder, word, letter_converter, phone_converter)
        predictions.append(output)

    return predictions

if __name__ == '__main__':
    # load models
    encoder, decoder, letter_converter, phone_converter = load_models()

    # load text
    text = open_file(DATA_PATH)

    # text into array
    words = get_words(text)

    # make predictions
    predictions = get_predictions(words, encoder, decoder, letter_converter, phone_converter)

    # save to file
    save_to_file(DATA_SAVE_PATH, words, predictions)

