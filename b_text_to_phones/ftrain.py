import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import config
import json
import dengine as engine
from torch.utils.data import TensorDataset, DataLoader
from utils.converter import Converter
import torch.optim.lr_scheduler as lr_scheduler
from cmodel import EncoderRNN, AttnDecoderRNN

DATA_JSON_PATH = './data_preprocessed/train_data_adjusted.json'
BUR_ALPHABET_PATH = './data_raw/bur_alphabet.txt'
BUR_PHONE_SET_PATH = 'data_raw/bur_phone_set.txt'
ECODER_SAVE_PATH = config.ECODER_SAVE_PATH
ATTN_DECODER_SAVE_PATH = config.ATTN_DECODER_SAVE_PATH


def load_data(fpath):
    with open(fpath) as fp:
        data = json.load(fp)
        return data['data']


def prepare_dataloader(data):
    num_pairs = len(data)
    input_ids = np.zeros((num_pairs, config.MAX_SEQUENCE_LENGTH), dtype=np.int32)
    target_ids = np.zeros((num_pairs, config.MAX_SEQUENCE_LENGTH), dtype=np.int32)

    for i, (inp, tgt) in enumerate(data):
        inp.append(config.EOS_TOKEN)
        tgt.append(config.EOS_TOKEN)
        input_ids[i, :len(inp)] = inp
        target_ids[i, :len(tgt)] = tgt

    train_data = TensorDataset(
        torch.LongTensor(input_ids).to(config.DEVICE),
        torch.LongTensor(target_ids).to(config.DEVICE)
    )

    train_dataloader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)

    return train_dataloader


def run_training():
    data = load_data(DATA_JSON_PATH)
    dataloader = prepare_dataloader(data)

    # get converters
    letter_converter = Converter(BUR_ALPHABET_PATH)
    phone_converter = Converter(BUR_PHONE_SET_PATH)

    encoder = EncoderRNN(
        input_vocab_size=letter_converter.n_elements,
        hidden_size=config.HIDDEN_SIZE
    ).to(config.DEVICE)

    # decoder = DecoderRNN(
    #     hidden_size=config.HIDDEN_SIZE,
    #     output_vocab_size=phone_converter.n_elements
    # ).to(config.DEVICE)
    attn_decoder = AttnDecoderRNN(
        hidden_size=config.HIDDEN_SIZE,
        output_vocab_size=phone_converter.n_elements
    ).to(config.DEVICE)

    encoder_optim = optim.Adam(encoder.parameters(), lr=5e-3)
    decoder_optim = optim.Adam(attn_decoder.parameters(), lr=5e-3)
    encoder_scheduler = lr_scheduler.LinearLR(encoder_optim, start_factor=1.0, end_factor=0.1, total_iters=config.EPOCHS)
    decoder_scheduler = lr_scheduler.LinearLR(decoder_optim, start_factor=1.0, end_factor=0.1, total_iters=config.EPOCHS)

    criterion = nn.NLLLoss()

    for epoch in range(1, config.EPOCHS + 1):
        loss = engine.train_fn(dataloader, encoder, attn_decoder, encoder_optim, decoder_optim, criterion)

        encoder_scheduler.step()
        decoder_scheduler.step()
        # if epoch % 5 == 0:
        # loss = engine.eval_fn(test_dataloader, encoder, decoder, criterion)

        if epoch % 5 == 0:
            print(f"Epoch: {epoch}, loss: {loss}")

    engine.save_model(encoder, ECODER_SAVE_PATH)
    engine.save_model(attn_decoder, ATTN_DECODER_SAVE_PATH)


if __name__ == '__main__':
    run_training()
