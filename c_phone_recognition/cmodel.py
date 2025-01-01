import torch
import torch.nn as nn
import utils.config as config
import torch.nn.functional as F

N_CHANNELS = 8
CONV_KERNEL_SIZE = (3, 1)
CONV_STRIDE = (1, 1)
POOL_KERNEL_SIZE = (2, 1)
POOL_STRIDE = (2, 1)
HIDDEN_SIZE = 128

class PhoneRecognitionModel(nn.Module):
    def __init__(self, output_size):
        super(PhoneRecognitionModel, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=N_CHANNELS, kernel_size=CONV_KERNEL_SIZE, stride=CONV_STRIDE),
            nn.Conv2d(in_channels=N_CHANNELS, out_channels=N_CHANNELS, kernel_size=CONV_KERNEL_SIZE, stride=CONV_STRIDE),
            nn.Conv2d(in_channels=N_CHANNELS, out_channels=N_CHANNELS, kernel_size=CONV_KERNEL_SIZE, stride=CONV_STRIDE),
            # nn.BatchNorm2d(N_CHANNELS),
            nn.ReLU(),
            nn.MaxPool2d(POOL_KERNEL_SIZE, POOL_STRIDE),
            nn.Conv2d(in_channels=N_CHANNELS, out_channels=N_CHANNELS, kernel_size=CONV_KERNEL_SIZE, stride=CONV_STRIDE),
            nn.Conv2d(in_channels=N_CHANNELS, out_channels=N_CHANNELS, kernel_size=CONV_KERNEL_SIZE, stride=CONV_STRIDE),
            nn.Conv2d(in_channels=N_CHANNELS, out_channels=N_CHANNELS, kernel_size=CONV_KERNEL_SIZE, stride=CONV_STRIDE),
            # nn.BatchNorm2d(N_CHANNELS),
            # nn.ReLU(),
            nn.MaxPool2d((4, 1), (4, 1)),
        )
        self.dropout = nn.Dropout(0.2)
        self.rnn = nn.GRU(
            # input_size=1856,
            input_size=104,
            hidden_size=HIDDEN_SIZE,
            batch_first=True
        )

        self.linear = nn.Linear(
            in_features=HIDDEN_SIZE,
            out_features=output_size
        )

    def forward(self, data, isTest=False):
        """
        :param input: bs, c, n_mels, ts
        :return:
        """
        bs, _, _, _ = data.size()

        # x: bs, c, n_mels, ts
        x = self.conv(data)

        if isTest:
            print('Current x.size():', x.size())

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



if __name__ == '__main__':
    n_mels = config.N_MELS
    # ts = 313 * 2 # (10 * config.SAMPLE_RATE - config.N_FFT) // config.HOP_LENGTH + 1
    ts = 307 # (10 * config.SAMPLE_RATE - config.N_FFT) // config.HOP_LENGTH + 1
    bs = 1
    c = 1
    data = torch.zeros((bs, c, n_mels, ts))
    print(data.size())
    model = PhoneRecognitionModel(output_size=10)
    out = model(data, isTest=True)
    print(out.size())



