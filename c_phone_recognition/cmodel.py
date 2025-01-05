import torch
import torch.nn as nn
import utils.config as config
import torch.nn.functional as F

N_CHANNELS = 8
CONV_KERNEL_SIZE = (3, 3)
CONV_STRIDE = (1, 1)
POOL_KERNEL_SIZE = (1, 1)
POOL_STRIDE = (2, 1)
HIDDEN_SIZE = 128

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


if __name__ == '__main__':
    n_mels = config.N_MELS
    ts = 307
    bs = 1
    c = 1
    data = torch.zeros((bs, c, n_mels, ts))
    print(data.size())
    model = PhoneRecognitionModel(output_size=10)
    out = model(data)
    print(out.size())
    print(model)
