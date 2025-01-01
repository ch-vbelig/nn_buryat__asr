import torch
import torch.nn as nn
import torch.nn.functional as F
import b_text_to_phones.utils.config as config

class EncoderRNN(nn.Module):
    def __init__(self, input_vocab_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_vocab_size, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        """
        :param input: (bs, MAX_SEQUENCE_LENGTH)
        :return: output: (bs, MAX_SEQUENCE_LENGTH, hidden_size)
        :return: hidden: (1, bs, hidden_size)
        """
        embedded = self.dropout(self.embedding(input))  # (bs, MAX_SEQUENCE_LENGTH, hidden_size)

        # get all outputs and the last hidden state
        # output: (bs, MAX_SEQUENCE_LENGTH, hidden_size)
        # hidden: (num_of_layers, bs, hidden_size) -> (1, bs, hidden_size)
        output, hidden = self.gru(embedded)

        return output, hidden



class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_vocab_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_vocab_size, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.out = nn.Linear(hidden_size, output_vocab_size)


    def forward(self, encoder_outputs, encoder_last_hidden, target_tensor=None):
        """
        :param encoder_outputs: all outputs from the encoder (in this case only for retrieving batch_size)
        -> (bs, MAX_SEQUENCE_LENGTH, hidden_size)
        :param encoder_last_hidden: context vector
        -> (1, bs, hidden_size)
        :param target_tensor: for Teacher forcing
        -> (bs, MAX_SEQUENCE_LENGTH)
        :return: decoder_outputs: (bs, MAX_SEQUENCE_LENGTH, output_size)
        :return: decoder_hidden: (1, bs, hidden_size)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bs = encoder_outputs.size(0)

        # Teacher Forcing:
        # initialize decoder_input with SOS token -> (bs, 1)
        SOS_token = 0
        decoder_input = torch.empty(bs, 1, dtype=torch.long, device=device).fill_(SOS_token)

        # initialize the decoder's hidden state with encoder's last hidden state -> (1, bs, hidden_size)
        decoder_hidden = encoder_last_hidden

        # store outputs
        decoder_outputs = []

        for i in range(config.MAX_SEQUENCE_LENGTH):
            # decoder_output: (bs, 1, output_size)
            # decoder_hidden: (1, bs, hidden_size)
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)

            # decoder_output: (N, bs, 1, output_size)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing -> feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                # feed its own predictions
                # (bs, 1, output_size) -> (bs, 1, 1)
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        # decoder_output: (N, bs, 1, output_size) -> (bs, MAX_SEQUENCE_LENGTH, output_size)
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)

        return decoder_outputs, decoder_hidden, None

    def forward_step(self, input, hidden):
        """
        :param input: (bs, 1)
        :param hidden: (1, bs, hidden_size)
        :return: decoder_output: the next token -> (bs, 1, output_size)
        :return: decoder_hidden: (1, bs, hidden_size)
        """

        output = self.embedding(input) # (bs, 1, hidden_size)
        output = F.relu(output)

        output, hidden = self.gru(output, hidden) # (bs, 1, hidden_size) and (1, bs, hidden_size)
        output = self.out(output)

        return output, hidden



class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys, idx):
        """
        Executed for every decoder's input word
        :param query: hidden state permuted (bs, 1, hidden_size) -> changes for every decoder's input
        :param keys: encoder_outputs (bs, MAX_SEQUENCE_LENGTH, hidden_size) -> stays the same
        :return: weights: # (bs, 1, MAX_SEQUENCE_LENGTH)
        :return: context: (bs, 1, hidden_size)
        """
        # MAX_LENGTH = keys.size(1)
        # LEFT_STEP = 2
        # RIGHT_STEP = 2
        # start = max(0, idx - LEFT_STEP)
        # end = min(MAX_LENGTH, idx + RIGHT_STEP)
        # start = 0
        # end = MAX_LENGTH

        # keys = keys[:, start:end, :]
        # print(keys.size())

        scores = self.Va(torch.tanh(
            self.Wa(query) + self.Ua(keys)
        )) # scores: (bs, MAX_SEQUENCE_LENGTH, 1)

        scores = scores.squeeze(2).unsqueeze(1) # scores: (bs, MAX_SEQUENCE_LENGTH, 1) -> (bs, 1, MAX_SEQUENCE_LENGTH)
        weights = F.softmax(scores, dim=-1) # (bs, 1, MAX_SEQUENCE_LENGTH)
        context = torch.bmm(weights, keys) # (bs, 1, hidden_size)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_vocab_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_vocab_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(
            input_size=2*hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.out = nn.Linear(hidden_size, output_vocab_size)
        self.dropout = nn.Dropout(dropout_p)
        self.idx = 0

    def forward(self, encoder_outputs, encoder_last_hidden, target_tensor=None):
        """
        :param encoder_outputs: all outputs from the encoder (in this case only for retrieving batch_size)
        -> (bs, MAX_SEQUENCE_LENGTH, hidden_size)
        :param encoder_last_hidden: context vector
        -> (1, bs, hidden_size)
        :param target_tensor: for Teacher forcing
        -> (bs, MAX_SEQUENCE_LENGTH)
        :return: decoder_outputs: (bs, MAX_SEQUENCE_LENGTH, output_size)
        :return: decoder_hidden: (1, bs, hidden_size)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bs = encoder_outputs.size(0)

        # Teacher Forcing:
        # initialize decoder_input with SOS token -> (bs, 1)
        SOS_token = 0
        decoder_input = torch.empty(bs, 1, dtype=torch.long, device=device).fill_(SOS_token)

        # initialize the decoder's hidden state with encoder's last hidden state -> (1, bs, hidden_size)
        decoder_hidden = encoder_last_hidden

        # store outputs
        decoder_outputs = []
        attentions = []

        for i in range(config.MAX_SEQUENCE_LENGTH):
            # decoder_output: (bs, 1, output_size)
            # decoder_hidden: (1, bs, hidden_size)
            # attn_weights: (bs, 1, MAX_SEQUENCE_LENGTH)
            decoder_output, decoder_hidden, attn_weights = self.forward_step(decoder_input, decoder_hidden, encoder_outputs, i)

            # decoder_outputs: (N, bs, 1, output_size)
            # attentions: (N, bs, 1, MAX_SEQUENCE_LENGTH)
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing -> feed the target as the next input
                # target_tensor[:, i]: from (bs, MAX_SEQUENCE LENGTH) -> (bs) -> (b1, 1)
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                # feed its own predictions
                # (bs, 1, output_size) -> (bs, 1, 1)
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        # decoder_output: (N, bs, 1, output_size) -> (bs, MAX_SEQUENCE_LENGTH, output_size)
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        # attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs, i):
        """
        :param input: (bs, 1)
        :param hidden: (1, bs, hidden_size)
        :return: decoder_output: the next token -> (bs, 1, output_size)
        :return: decoder_hidden: (1, bs, hidden_size)
        :return: attn_weights: (bs, 1, MAX_SEQUENCE_LENGTH)
        """

        # bs, 1, hidden_size
        embedded = self.dropout(self.embedding(input)) # (bs, 1, hidden_size)

        query = hidden.permute(1, 0, 2) # -> (bs, 1, hidden_size)
        # context: (bs, 1, hidden_size)
        # weights: # (bs, 1, MAX_SEQUENCE_LENGTH)
        context, attn_weights = self.attention(query, encoder_outputs, i)

        # embedded: bs, 1, hidden_size
        # context: (bs, 1, hidden_size)
        # input_gru: (bs, 1, 2*hidden_size)
        input_gru = torch.cat((embedded, context), dim=2)

        # output: (bs, 1, hidden_size)
        # hidden: (1, bs, hidden_size)
        output, hidden = self.gru(input_gru, hidden)

        # output: (bs, 1, output_size)
        output = self.out(output)

        # output: (bs, 1, output_size)
        # hidden: (1, bs, hidden_size)
        # attn_weights: (bs, 1, MAX_SEQUENCE_LENGTH)
        return output, hidden, attn_weights



if __name__ == '__main__':
    batch_size = 5
    input_vocab_size = 10
    output_vocab_size = 11
    hidden_size = 64
    device = "cuda"

    encoder = EncoderRNN(input_vocab_size, hidden_size, 0.1).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_vocab_size).to(device)

    data = torch.zeros((batch_size, config.MAX_SEQUENCE_LENGTH), dtype=torch.long).to(device)

    out, hid = encoder(data)
    print(out.size())
    print(hid.size())
    # torch.Size([5, 10, 64])
    # torch.Size([1, 5, 64])

    out, hid, _ = decoder(out, hid)
    print(out.size())
    print(hid.size())
    # torch.Size([5, 10, 11])
    # torch.Size([1, 5, 64])
