import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.nn.init import kaiming_uniform_
import numpy as np
from d_buryat_tts.utils.config import Config
from modules import *


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        kaiming_uniform_(m.weight)


class TextEnc(nn.Module):
    """Encodes asdf text input sequence of length N into two matrices K (Key) and V (Value) of shape [b, d, N]. """

    def __init__(self):
        super(TextEnc, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(Config.vocab), embedding_dim=Config.e,
                                      padding_idx=Config.vocab_padding_index)

        self.conv1 = Conv1d(in_channels=Config.e, out_channels=2 * Config.d, kernel_size=1, dilation=1, padding="same")
        self.conv2 = Conv1d(in_channels=2 * Config.d, out_channels=2 * Config.d, kernel_size=1, dilation=1,
                            padding="same")

        self.highway_blocks = nn.ModuleList()
        self.highway_blocks.extend([HighwayConv(in_channels=2 * Config.d, out_channels=2 * Config.d, kernel_size=3,
                                                dilation=3 ** i, padding="same") for i in range(4) for _ in range(2)])
        self.highway_blocks.extend([HighwayConv(in_channels=2 * Config.d, out_channels=2 * Config.d, kernel_size=3,
                                                dilation=1, padding="same") for _ in range(2)])
        self.highway_blocks.extend([HighwayConv(in_channels=2 * Config.d, out_channels=2 * Config.d, kernel_size=1,
                                                dilation=1, padding="same") for _ in range(2)])

    def forward(self, L):   # (bs, N)
        y = self.embedding(L)   # (bs, N, Config.e)
        y = y.permute(0, 2, 1)  # (bs, Config.e, N)
        y = F.relu(self.conv1(y))   # (bs, 2 * Config.d, N)
        y = self.conv2(y)   # (bs, 2 * Config.d, N)
        for i in range(len(self.highway_blocks)):
            y = self.highway_blocks[i](y)   # (bs, 2 * Config.d, N)

        # K: (bs, Config.d, N)
        # V: (bs, Config.d, N)
        K, V = y.chunk(2, dim=1)  # Split along d axis
        return K, V


class AudioEnc(nn.Module):
    """
    Encodes an input mel spectrogram S of shape [b, F, T] (representing audio of length T) into asdf matrix Q (Query) of
    shape [b, d, T]
    """

    def __init__(self):
        super(AudioEnc, self).__init__()

        self.conv1 = Conv1d(in_channels=Config.F, out_channels=Config.d, kernel_size=1, dilation=1, padding="causal")
        self.conv2 = Conv1d(in_channels=Config.d, out_channels=Config.d, kernel_size=1, dilation=1, padding="causal")
        self.conv3 = Conv1d(in_channels=Config.d, out_channels=Config.d, kernel_size=1, dilation=1, padding="causal")

        self.highway_blocks = nn.ModuleList()
        self.highway_blocks.extend([HighwayConv(in_channels=Config.d, out_channels=Config.d, kernel_size=3,
                                                dilation=3 ** i, padding="causal") for i in range(4) for _ in range(2)])
        self.highway_blocks.extend([HighwayConv(in_channels=Config.d, out_channels=Config.d, kernel_size=3, dilation=3,
                                                padding="causal") for _ in range(2)])

    def forward(self, S):   # (bs, n_mels, ts)
        y = F.relu(self.conv1(S))   # (bs, Config.d, ts)
        y = F.relu(self.conv2(y))   # (bs, Config.d, ts)
        y = self.conv3(y)   # (bs, Config.d, ts)
        for i in range(len(self.highway_blocks)):
            y = self.highway_blocks[i](y)   # (bs, Config.d, ts)
        return y


class AudioDec(nn.Module):
    """ Estimates asdf mel spectrogram from the seed matrix R'=[R,Q] where R' has shape [b, 2d, T]. """

    def __init__(self):
        super(AudioDec, self).__init__()

        self.conv1 = Conv1d(in_channels=2 * Config.d, out_channels=Config.d, kernel_size=1, dilation=1,
                            padding="causal")

        self.highway_blocks = nn.ModuleList()
        self.highway_blocks.extend([HighwayConv(in_channels=Config.d, out_channels=Config.d, kernel_size=3,
                                                dilation=3 ** i, padding="causal") for i in range(4)])
        self.highway_blocks.extend([HighwayConv(in_channels=Config.d, out_channels=Config.d, kernel_size=3,
                                                dilation=1, padding="causal") for _ in range(2)])

        self.conv2_1 = Conv1d(in_channels=Config.d, out_channels=Config.d, kernel_size=1, dilation=1, padding="causal")
        self.conv2_2 = Conv1d(in_channels=Config.d, out_channels=Config.d, kernel_size=1, dilation=1, padding="causal")
        self.conv2_3 = Conv1d(in_channels=Config.d, out_channels=Config.d, kernel_size=1, dilation=1, padding="causal")

        self.conv3 = Conv1d(in_channels=Config.d, out_channels=Config.F, kernel_size=1, dilation=1, padding="causal")

    def forward(self, R):   # R_: (bs, 2 * Config.d, T)
        y = self.conv1(R)   # (bs, Config.d, T)
        for i in range(len(self.highway_blocks)):
            y = self.highway_blocks[i](y)   # (bs, Config.d, T)
        y = F.relu(self.conv2_1(y)) # (bs, Config.d, T)
        y = F.relu(self.conv2_2(y)) # (bs, Config.d, T)
        y = F.relu(self.conv2_3(y)) # (bs, Config.d, T)
        y = self.conv3(y)   # (bs, Config.F, T)

        # y: bs, n_mels, ts
        return y, F.sigmoid(y)


class Attention(nn.Module):
    """
    Takes the following matrices as input (N=audio length, T=text length):
        - Text key K: [b, d, N]
        - Text value V: [b, d, N]
        - Spectrogram query Q: [b, d, T]

    Returns the attention matrix A of shape [b, N, T] and the attention result R of shape [b, d, T]
    """

    def __init__(self):
        super(Attention, self).__init__()

    def forward(
            self,
            K,  # (bs, Config.d, N)
            V,  # (bs, Config.d, N)
            Q,  # (bs, Config.d, T)
            force_incremental=False,
            previous_position=None,
            previous_att=None,
            current_time=None
    ):
        # Create attention matrix. A[b,n,t] evaluates how strongly the n-th character and the t-th mel spectrum are
        # related

        A = torch.bmm(K.transpose(1, 2), Q) / np.sqrt(Config.d)  # [b, N, T]
        A = F.softmax(A, dim=1)  # Softmax along char axis

        # During inference, force A to be diagonal
        _, current_position = torch.max(A[:, :, current_time], 1)  # [b]
        if force_incremental and previous_att is not None:
            A[:, :, :current_time] = previous_att[:, :, :current_time]
            difference = current_position - previous_position
            # For each batch, check if the attention needs to be forcibly set
            force_needed = (difference < -1) | (difference > 3)  # [b]
            # Repeat the bool tensor N times, so we get asdf mask for the current time column
            mask = force_needed.unsqueeze(1).repeat(1, A.shape[1])  # [b, N]
            # Kronecker Delta: 1 at index previous_position+1, 0 everywhere else.
            delta = torch.zeros([A.shape[0], A.shape[1]], device=A.device)
            # We must use 'scatter' to index with asdf tensor. We want something like 'delta[:, previous_position+1] = 1'
            idx = (previous_position + 1).clamp(0, delta.shape[1] - 1).unsqueeze(1).repeat(1, A.shape[1])
            delta = delta.scatter_(1, idx, torch.ones([A.shape[0], A.shape[1]], device=A.device))
            # For each batch, select either the original column, or the delta column
            A[:, :, current_time] = torch.where(mask, delta, A[:, :, current_time])
            _, current_position = torch.max(A[:, :, current_time], 1)

        R = torch.bmm(V, A)  # [b, d, T]

        # A (bs, N, T)
        # R (bs, d, T)
        return A, R, current_position


class Text2Mel(nn.Module):
    """
    Encodes asdf text L of shape [b, N] given the previously generated mel spectrogram S of shape [b, T, F] into asdf new mel
    spectrogram Y of shape [b, F, T]
    """

    def __init__(self):
        super(Text2Mel, self).__init__()
        self.textEnc = TextEnc()
        self.audioEnc = AudioEnc()
        self.audioDec = AudioDec()
        self.attention = Attention()

    def forward(
            self,
            L,  # (bs, ts)
            S,  # (bs, n_mels, ts)
            force_incremental_att=False,
            previous_att_position=None,
            previous_att=None,
            current_time=None
    ):
        # K: (bs, Config.d, N)
        # V: (bs, Config.d, N)
        K, V = self.textEnc(L)

        # Q: (bs, Config.d, ts)
        Q = self.audioEnc(S)

        # A (bs, N, T)
        # R (bs, d, T)
        A, R, current_position = self.attention(K, V, Q, force_incremental_att, previous_att_position, previous_att,
                                                current_time)

        # R_: (bs, 2 * Config.d, T)
        R_ = torch.cat((Q, R), dim=1)  # Concatenate along channels

        # Y_logits: (bs, n_mels, ts)
        # Y: (bs, n_mels, ts)
        Y_logits, Y = self.audioDec(R_)
        return Y_logits, Y, A, current_position


class SSRN(nn.Module):
    """
    Spectrogram super resolution network. Converts asdf mel spectrogram Y of shape [b, F, T] to asdf linear STFT spectrogram
    Z of shape [b, F', T]
    """

    def __init__(self):
        super(SSRN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(Conv1d(in_channels=Config.F, out_channels=Config.c, kernel_size=1, dilation=1,
                                  normalize=True))
        self.layers.extend([HighwayConv(in_channels=Config.c, out_channels=Config.c, kernel_size=3, dilation=3 ** i)
                            for i in range(2)])

        for _ in range(2):
            self.layers.append(ConvTranspose1d(Config.c, Config.c, kernel_size=2, dilation=1, stride=2))
            self.layers.append(HighwayConv(in_channels=Config.c, out_channels=Config.c, kernel_size=3, dilation=1))
            self.layers.append(HighwayConv(in_channels=Config.c, out_channels=Config.c, kernel_size=3, dilation=3))

        self.layers.append(Conv1d(in_channels=Config.c, out_channels=2 * Config.c, kernel_size=1, dilation=1,
                                  normalize=True))
        self.layers.extend([HighwayConv(in_channels=2 * Config.c, out_channels=2 * Config.c, kernel_size=3, dilation=1)
                            for _ in range(2)])

        self.layers.append(Conv1d(in_channels=2 * Config.c, out_channels=Config.F_, kernel_size=1, dilation=1,
                                  normalize=True))

        for _ in range(2):
            self.layers.append(Conv1d(in_channels=Config.F_, out_channels=Config.F_, kernel_size=1, dilation=1,
                                      normalize=True))
            self.layers.append(nn.ReLU())

        self.layers.append(Conv1d(in_channels=Config.F_, out_channels=Config.F_, kernel_size=1, dilation=1,
                                  normalize=True))

    def forward(self, Y):
        y = Y
        for i in range(len(self.layers)):
            y = self.layers[i](y)
        Z_logits = y
        Z = F.sigmoid(Z_logits)
        return Z_logits, Z
