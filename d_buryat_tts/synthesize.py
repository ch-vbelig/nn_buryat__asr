import argparse
from models import SSRN, Text2Mel
from d_buryat_tts.utils.config import Config
from d_buryat_tts.utils.text_processing import *
from d_buryat_tts.utils.audio_processing import spectrogram2wav, vocoder
from dataset import *
import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import librosa


parser = argparse.ArgumentParser()
parser.add_argument('--t2m', dest='text2mel_path', required=True, help='Path to Text2Mel save file')
parser.add_argument('--ssrn', dest='ssrn_path', required=True, help='Path to SSRN save file')


if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    assert os.path.exists(args.text2mel_path), "File '{}' does not exist!".format(args.text2mel_path)
    assert os.path.exists(args.ssrn_path), "File '{}' does not exist!".format(args.ssrn_path)

    # Restore config
    state_t2m = torch.load(args.text2mel_path, map_location=device)
    config_t2m = state_t2m["config"]
    state_ssrn = torch.load(args.ssrn_path, map_location=device)
    config_ssrn = state_ssrn["config"]
    if config_ssrn != config_t2m:
        print("WARNING: Text2Mel and SSRN have different saved configs. Will use Text2Mel config!")
    # Config.set_config(config_t2m)

    # Load networks
    print("Loading Text2Mel...")
    text2mel = Text2Mel().to(device)
    text2mel.eval()
    text2mel_step = state_t2m["global_step"]
    text2mel.load_state_dict(state_t2m["model"])

    print("Loading SSRN...")
    ssrn = SSRN().to(device)
    ssrn.eval()
    ssrn_step = state_ssrn["global_step"]
    ssrn.load_state_dict(state_ssrn["model"])

    text = "би энэ сугтаа хүдэлһэн нүхэр тухай хоорэхэ гэжэ ерээд һуужа байнаб тиигээд лэ но намтай наһаарааш сасуу гэхэдэ болохолда намһаа нэгэ аха дүшэ долоо ондо түрэһэн агын гарбалтай"
    text = normalize(text)
    text = text + Config.vocab_end_of_text
    text = vocab_lookup(text)

    L = torch.tensor(text, device=device, requires_grad=False).unsqueeze(0)
    S = torch.zeros(1, Config.max_T, Config.F, requires_grad=False, device=device)  # S: (bs, T, n_mels)
    previous_position = torch.zeros(1, requires_grad=False, dtype=torch.long, device=device) # tensor([0]) # (1)
    previous_att = torch.zeros(1, len(text), Config.max_T, requires_grad=False, device=device) # 1, N, max_T

    print(S.size())
    for t in range(Config.max_T-1):
        # Y: (bs, n_mels, ts)
        _, Y, A, current_position = text2mel.forward(L, S.transpose(1, 2),
                                                     force_incremental_att=True,
                                                     previous_att_position=previous_position,
                                                     previous_att=previous_att,
                                                     current_time=t)
        # S: (bs, T, n_mels)
        # Y (transposed): (bs, T, n_mels)
        Y = Y.transpose(1, 2).detach()[:, t, :]
        S[:, t + 1, :] = Y
        previous_position = current_position.detach()
        previous_att = A.detach()

    mel = S.detach().cpu().numpy()
    mel = mel.squeeze()
    mel = mel.T



    # Generate linear spectrogram.
    _, Z = ssrn.forward(S.transpose(1, 2))
    Z = Z.transpose(1, 2).detach().cpu().numpy()
    wav = spectrogram2wav(Z[0])
    wav = np.concatenate([np.zeros(10000), wav], axis=0)  # Silence at the beginning
    wav *= 32767 / max(abs(wav))
    wav = wav.astype(np.int16)
    wav = torch.tensor(wav)
    wav = wav.unsqueeze(0)

    # wav_vocoder = vocoder(S.transpose(1, 2))
    # print(wav_vocoder.size())

    # fig, ax = plt.subplots(nrows=2)
    # img = librosa.display.specshow(data=mel, x_axis='time', y_axis='hz')
    # fig.colorbar(img, ax=ax[0], format="%+2.0f dB")

    # ax[1].plot(wav[0].numpy())

    fig, ax = plt.subplots()
    img = ax.imshow(A.detach().cpu()[0])
    fig.colorbar(img)
    plt.show()

    torchaudio.save(
        uri='synthesized_speech/speech_8.wav',
        src=wav,
        sample_rate=Config.sample_rate
    )


