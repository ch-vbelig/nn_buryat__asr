import b_text_to_phones.dengine as engine
from b_text_to_phones.utils.converter import Converter
from b_text_to_phones.cmodel import EncoderRNN, AttnDecoderRNN
import b_text_to_phones.utils.config as config

if __name__ == '__main__':
    BUR_ALPHABET_PATH = '../data_raw/bur_alphabet.txt'
    BUR_PHONE_SET_PATH = '../data_raw/bur_phone_set.txt'
    ENCODER_PATH = '../model/encoder.pth'
    ATTN_DECODER_PATH = '../model/attn_decoder.pth'

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

    word = 'шэлэдэгдэхээр'
    output = engine.predict(encoder, decoder, word, letter_converter, phone_converter)

    print(output[:-1])


