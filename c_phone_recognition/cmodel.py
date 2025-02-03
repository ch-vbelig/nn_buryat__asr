import torch
import torchaudio
import torch.nn as nn
import utils.config as config
import torch.nn.functional as F

N_CHANNELS = 16
CONV_KERNEL_SIZE = (3, 3)
CONV_STRIDE = (1, 1)
PADDING_SIZE = (CONV_KERNEL_SIZE[0] // 2, CONV_KERNEL_SIZE[1] // 2)
POOL_KERNEL_SIZE = (3, 1)
POOL_STRIDE = (2, 1)
HIDDEN_SIZE = 128


class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=N_CHANNELS, out_channels=N_CHANNELS, kernel_size=CONV_KERNEL_SIZE,
                      stride=CONV_STRIDE, padding=PADDING_SIZE),
            nn.Conv2d(in_channels=N_CHANNELS, out_channels=N_CHANNELS, kernel_size=CONV_KERNEL_SIZE,
                      stride=CONV_STRIDE, padding=PADDING_SIZE),
            nn.Conv2d(in_channels=N_CHANNELS, out_channels=N_CHANNELS, kernel_size=CONV_KERNEL_SIZE,
                      stride=CONV_STRIDE, padding=PADDING_SIZE),
            nn.BatchNorm2d(N_CHANNELS),
        )
        self.relu = nn.ReLU()

    def forward(self, data):
        residual = data
        x = self.conv(data)
        x = x + residual
        x = self.relu(x)
        return x


class PhoneRecognitionModelResidual(nn.Module):
    def __init__(self, output_size):
        super(PhoneRecognitionModelResidual, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=N_CHANNELS, kernel_size=CONV_KERNEL_SIZE, stride=CONV_STRIDE, padding=2),
            ResidualBlock(),
            nn.MaxPool2d(POOL_KERNEL_SIZE, POOL_STRIDE),
            ResidualBlock(),
            nn.MaxPool2d(POOL_KERNEL_SIZE, POOL_STRIDE),
        )
        self.dropout = nn.Dropout(0.2)
        rnn_input_size = self._get_cnn_out_size()
        self.rnn = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=HIDDEN_SIZE,
            batch_first=True,
        )

        self.linear = nn.Linear(
            in_features=HIDDEN_SIZE,
            out_features=output_size
        )

    def forward(self, data):
        """
        :param input: bs, c, n_mels, ts
        :return:
        """
        bs, _, _, _ = data.size()

        # x: bs, c, n_mels, ts
        x = self.conv(data)

        # x: bs, ts, c, n_mels
        x = x.permute(0, 3, 1, 2)

        # x: bs, ts, c * n_mels
        x = x.view(bs, x.size(1), -1)

        x = self.dropout(x)

        # outs: bs, ts, hidden_size
        outs, _ = self.rnn(x)

        # outs: bs, ts, output_size
        outs = self.linear(outs)

        # outs: ts, bs, output_size
        outs = outs.permute(1, 0, 2)

        log_probs = F.log_softmax(outs, dim=2)

        return log_probs

    def _get_cnn_out_size(self, verbose=False):
        # Configs for mock_data
        bs, n_channels, n_mels, ts = (
            1,
            1,
            config.N_MELS,
            (10 * config.SAMPLE_RATE - config.N_FFT) // config.HOP_LENGTH + 1)

        mock_data = torch.zeros((bs, n_channels, n_mels, ts), requires_grad=False)

        size_before = mock_data.size()

        # Get output for the convolution layer
        out = self.conv(mock_data)
        size_after = out.size()

        if verbose:
            print(f"Before: {size_before}")
            print(f"After: {size_after}")

        # Calculate input size for the rnn layer
        n_channels = size_after[1]
        n_mels = size_after[2]

        n_neurons = n_channels * n_mels

        return n_neurons


class PhoneRecognitionModel(nn.Module):
    def __init__(self, output_size):
        super(PhoneRecognitionModel, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=N_CHANNELS, kernel_size=CONV_KERNEL_SIZE, stride=CONV_STRIDE),
            nn.Conv2d(in_channels=N_CHANNELS, out_channels=N_CHANNELS, kernel_size=CONV_KERNEL_SIZE,
                      stride=CONV_STRIDE),
            nn.Conv2d(in_channels=N_CHANNELS, out_channels=N_CHANNELS, kernel_size=CONV_KERNEL_SIZE,
                      stride=CONV_STRIDE),
            nn.BatchNorm2d(N_CHANNELS),
            nn.ReLU(),
            nn.MaxPool2d(POOL_KERNEL_SIZE, POOL_STRIDE),
            nn.Conv2d(in_channels=N_CHANNELS, out_channels=N_CHANNELS, kernel_size=CONV_KERNEL_SIZE,
                      stride=CONV_STRIDE),
            nn.Conv2d(in_channels=N_CHANNELS, out_channels=N_CHANNELS, kernel_size=CONV_KERNEL_SIZE,
                      stride=CONV_STRIDE),
            nn.Conv2d(in_channels=N_CHANNELS, out_channels=N_CHANNELS, kernel_size=CONV_KERNEL_SIZE,
                      stride=CONV_STRIDE),
            nn.BatchNorm2d(N_CHANNELS),
            nn.ReLU(),
            nn.MaxPool2d(POOL_KERNEL_SIZE, POOL_STRIDE),
        )
        self.dropout = nn.Dropout(0.2)
        rnn_input_size = self._get_cnn_out_size()
        self.rnn = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=HIDDEN_SIZE,
            batch_first=True,
        )

        self.linear = nn.Linear(
            in_features=HIDDEN_SIZE,
            out_features=output_size
        )

    def forward(self, data):
        """
        :param input: bs, c, n_mels, ts
        :return:
        """
        bs, _, _, _ = data.size()

        # x: bs, c, n_mels, ts
        x = self.conv(data)

        # x: bs, ts, c, n_mels
        x = x.permute(0, 3, 1, 2)

        # x: bs, ts, c * n_mels
        x = x.view(bs, x.size(1), -1)

        x = self.dropout(x)

        # outs: bs, ts, hidden_size
        outs, _ = self.rnn(x)

        # outs: bs, ts, output_size
        outs = self.linear(outs)

        # outs: ts, bs, output_size
        outs = outs.permute(1, 0, 2)

        log_probs = F.log_softmax(outs, dim=2)

        return log_probs

    def _get_cnn_out_size(self, verbose=False):
        # Configs for mock_data
        bs, n_channels, n_mels, ts = (
            1,
            1,
            config.N_MELS,
            (10 * config.SAMPLE_RATE - config.N_FFT) // config.HOP_LENGTH + 1)

        mock_data = torch.zeros((bs, n_channels, n_mels, ts), requires_grad=False)

        size_before = mock_data.size()

        # Get output for the convolution layer
        out = self.conv(mock_data)
        size_after = out.size()

        if verbose:
            print(f"Before: {size_before}")
            print(f"After: {size_after}")

        # Calculate input size for the rnn layer
        n_channels = size_after[1]
        n_mels = size_after[2]

        n_neurons = n_channels * n_mels

        return n_neurons


class PhoneRecognitionModelCnn(nn.Module):
    def __init__(self, output_size):
        super(PhoneRecognitionModelCnn, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=N_CHANNELS, kernel_size=(20, 1), stride=(1, 1)),
            nn.Conv2d(in_channels=N_CHANNELS, out_channels=N_CHANNELS, kernel_size=(10, 1), stride=(1, 1)),
            nn.Conv2d(in_channels=N_CHANNELS, out_channels=N_CHANNELS, kernel_size=(7, 1), stride=(1, 1)),
            nn.Conv2d(in_channels=N_CHANNELS, out_channels=N_CHANNELS, kernel_size=(7, 1), stride=(1, 1)),
            nn.Conv2d(in_channels=N_CHANNELS, out_channels=N_CHANNELS, kernel_size=(7, 1), stride=(1, 1)),
            nn.Conv2d(in_channels=N_CHANNELS, out_channels=N_CHANNELS, kernel_size=(7, 1), stride=(1, 1)),
            nn.Conv2d(in_channels=N_CHANNELS, out_channels=1, kernel_size=(1, 1), stride=(1, 1)),
        )
        n_neurons = self._get_cnn_out_size()
        self.linear = nn.Linear(
            in_features=n_neurons,
            out_features=output_size
        )

    def forward(self, data):
        """
        :param input: bs, c, n_mels, ts
        :return:
        """
        bs, _, _, _ = data.size()

        # x: bs, c, n_mels, ts
        x = self.conv(data)
        # x: bs, ts, c, n_mels
        x = x.permute(0, 3, 1, 2)
        x = self.linear(x)
        x = x.view(x.size(1), -1, x.size(3))

        out = F.log_softmax(x, dim=2)

        return out

    def _get_cnn_out_size(self, verbose=False):
        # Configs for mock_data
        bs, n_channels, n_mels, ts = (
            1,
            1,
            config.N_MELS,
            (10 * config.SAMPLE_RATE - config.N_FFT) // config.HOP_LENGTH + 1)

        mock_data = torch.zeros((bs, n_channels, n_mels, ts), requires_grad=False)

        size_before = mock_data.size()

        # Get output for the convolution layer
        out = self.conv(mock_data)
        size_after = out.size()

        if verbose:
            print(f"Before: {size_before}")
            print(f"After: {size_after}")

        # Calculate input size for the rnn layer
        n_channels = size_after[1]
        n_mels = size_after[2]

        n_neurons = n_channels * n_mels

        return n_neurons


class PhoneRecognitionModelWav2Vec(nn.Module):
    def __init__(self, output_size):
        super(PhoneRecognitionModelWav2Vec, self).__init__()
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

        self.wav2vec = bundle.get_model()

        self.wav2vec.aux = nn.Linear(
            in_features=self.wav2vec.aux.in_features,
            out_features=output_size,
            bias=True
        )

    def freeze_wav2vec(self):
        for parameter in self.wav2vec.parameters():
            parameter.requires_grad = False
        for parameter in self.wav2vec.aux.parameters():
            parameter.requires_grad = True

    def forward(self, waveform):
        """
        :param input: bs, 1, ts
        :return:
        """
        outs, _ = self.wav2vec(waveform)

        log_probs = F.log_softmax(outs, dim=2)
        log_probs = log_probs.permute(1, 0, 2)

        return log_probs


if __name__ == '__main__':
    n_mels = config.N_MELS
    ts = 307
    bs = 4
    c = 1
    data = torch.zeros((bs, c, n_mels, ts))
    # print(data.size())
    model = PhoneRecognitionModelResidual(output_size=10)
    out = model(data)
    print(out.size())
    # print(model)

    # path = './data/test_audio/test3.wav'
    # waveform, sr = torchaudio.load(path)
    # # waveform = waveform.unsqueeze(0)
    # print(waveform.size())
    #
    # model = PhoneRecognitionModelWav2Vec(61)
    # emission = model(waveform)

    # model = PhoneRecognitionModelCnn(output_size=10)
    # out = model(data)
    # print(out.size())
