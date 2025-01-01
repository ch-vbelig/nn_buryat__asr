import torch
import numpy as np
import b_text_to_phones.utils.config as config
from b_text_to_phones.cmodel import EncoderRNN, DecoderRNN
from b_text_to_phones.utils.converter import Converter
import re

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path, weights_only=True))
    return model

def train_fn(
        data_loader,
        encoder: EncoderRNN,
        decoder: DecoderRNN,
        encoder_optim,
        decoder_optim,
        criterion
    ):
    encoder.train()
    decoder.train()

    total_loss = 0

    for data in data_loader:
        input_tensor, target_tensor = data
        input_tensor.to("cuda")
        target_tensor.to("cuda")

        encoder_optim.zero_grad()
        decoder_optim.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            # (bs, MAX_SEQUENCE_LENGTH, output_size) -> (bs x MAX_SEQUENCE_LENGTH, output_size)
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            # (bs, MAX_SEQUENCE_LENGTH) -> (bs x MAX_SEQUENCE_LENGTH)
            target_tensor.view(-1)
        )

        loss.backward()

        encoder_optim.step()
        decoder_optim.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def eval_fn(
        data_loader,
        encoder: EncoderRNN,
        decoder: DecoderRNN,
        criterion
):
    encoder.eval()
    decoder.eval()

    total_loss = 0

    with torch.no_grad():
        for data in data_loader:
            input_tensor, target_tensor = data
            input_tensor.to("cuda")
            target_tensor.to("cuda")

            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

            loss = criterion(
                # (bs, MAX_SEQUENCE_LENGTH, output_size) -> (bs x MAX_SEQUENCE_LENGTH, output_size)
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                # (bs, MAX_SEQUENCE_LENGTH) -> (bs x MAX_SEQUENCE_LENGTH)
                target_tensor.view(-1)
            )

        total_loss += loss.item()

    return total_loss / len(data_loader)

def predict(encoder: EncoderRNN, decoder: DecoderRNN, word, letter_converter: Converter, phone_converter: Converter):
    encoder.eval()
    decoder.eval()

    encoder.to(config.DEVICE)
    decoder.to(config.DEVICE)

    word = _normalize_word(word)

    idxs = np.zeros((1, config.MAX_SEQUENCE_LENGTH), dtype=np.int32)

    # convert word into idxs
    idxs_from_word = _to_index(word, letter_converter.el_to_index)
    idxs[0, :len(idxs_from_word)] = idxs_from_word

    idxs = torch.LongTensor(idxs).to(config.DEVICE)

    with torch.no_grad():
        encoder_outputs, encoder_hidden = encoder(idxs)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden)

        # (bs, MAX_SEQUENCE_LENGTH, output_size) -> (bs, MAX_SEQUENCE_LENGTH, 1)
        # (1, MAX_SEQUENCE_LENGTH, output_size) -> (1, MAX_SEQUENCE_LENGTH, 1)
        _, topi = decoder_outputs.topk(1)

        # (MAX_SEQUENCE_LENGTH)
        output_idxs = topi.view(-1).to('cpu').numpy()

        output_phones = [phone_converter.index_to_el[idx] for idx in output_idxs if idx > 0]

        return output_phones

def _to_index(word, to_index):
    return [to_index[l] for l in list(word)]


def _normalize_word(word):
    word = word.lower()
    word = re.sub('\W', '', word)
    word = re.sub('\d', '', word)
    return word