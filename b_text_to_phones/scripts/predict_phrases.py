import b_text_to_phones.dengine as engine
from b_text_to_phones.utils.converter import Converter
from b_text_to_phones.cmodel import EncoderRNN, AttnDecoderRNN
import b_text_to_phones.utils.config as config
import re

BUR_ALPHABET_PATH = '../data_raw/bur_alphabet.txt'
BUR_PHONE_SET_PATH = '../data_raw/bur_phone_set.txt'
DATA_PATH = '../data_raw/predict_phrases.txt'
DATA_SAVE_PATH = '../data_raw/predicted_phones_for_phrases.txt'
ENCODER_PATH = '../model/encoder.pth'
ATTN_DECODER_PATH = '../model/attn_decoder.pth'

def open_file(fpath):
    with open(fpath, encoding='utf-8') as fp:
        return fp.read()

def save_to_file(fpath, words, predictions):
    with open(fpath, 'w', encoding='utf-8') as fp:
        for sentence, phones in zip(words, predictions):
            word_line = ' '.join(sentence)
            phone_line = ' <SPACE> '.join(phones)
            # line = f'{word_line}\t{phone_line}\n'
            line = f'{phone_line}\n'
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

    encoder = engine.load_model(encoder, ENCODER_PATH)
    decoder = engine.load_model(decoder, ATTN_DECODER_PATH)

    return encoder, decoder, letter_converter, phone_converter

def get_sentences(text):
    sentences = []
    lines = [l.lower().strip() for l in text.split('\n')]
    for line in lines:
        words = [w for w in line.split()]
        words = [w for w in words if re.match('\w+', w)]
        words = [w for w in words if re.match('\D+', w)]
        sentences.append(words)
    return sentences

def get_phones_for_line(words, encoder, decoder, letter_converter, phone_converter):
    phones = []

    for word in words:
        output = engine.predict(encoder, decoder, word, letter_converter, phone_converter)
        phones.append(output[:-1])

    return phones

def predict_phrases(sentences, encoder, decoder, letter_converter, phone_converter):
    predictions = []

    for words in sentences:
        phones = get_phones_for_line(words, encoder, decoder, letter_converter, phone_converter)
        phones_str = [" ".join(phone_seq) for phone_seq in phones]
        predictions.append(phones_str)

    return predictions

if __name__ == '__main__':
    # load models
    encoder, decoder, letter_converter, phone_converter = load_models()

    # load text
    text = open_file(DATA_PATH)

    # text into array
    sentences = get_sentences(text)

    # make predictions
    predictions = predict_phrases(sentences, encoder, decoder, letter_converter, phone_converter)

    print(sentences)
    print(predictions)

    # save to file
    save_to_file(DATA_SAVE_PATH, sentences, predictions)

