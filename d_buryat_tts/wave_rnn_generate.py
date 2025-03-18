"""WaveRNN generation"""

import argparse
import os

import numpy as np
import soundfile as sf
import torch

from utils.config import Config as cfg
from models import WaveRNN


def generate(checkpoint_path, out_dir):
    """Generate waveforms from mel-spectrograms using WaveRNN
    """
    os.makedirs(out_dir, exist_ok=True)

    # Specify the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model
    model = WaveRNN()
    model = model.to(device)
    model.eval()

    checkpoint = torch.load(checkpoint_path,
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["model"])
    model_step = checkpoint["step"]

    mel_file = "C:\\Users\\buryat_saram\\Music\\Project Buryat Saram\\buryat_text_to_speech\\data_specs_mels\\reading_0_0.npy"

    mel = np.load(mel_file) # (ts, 80)
    mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        wav_hat = model.generate(mel)

    out_path = os.path.join(out_dir,
                            f"model_step{model_step:09d}_generated.wav")

    sf.write(out_path, wav_hat, cfg.sample_rate)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate waveforms from mel-spectrograms using WaveRNN")

    parser.add_argument(
        "--checkpoint_path",
        help="Path to the checkpoint to use to instantiate the model",
        required=True)

    parser.add_argument(
        "--out_dir",
        help="Path to the dir where generated waveforms will be saved",
        default="synthesized_speech")

    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    out_dir = args.out_dir

    generate(checkpoint_path, out_dir)