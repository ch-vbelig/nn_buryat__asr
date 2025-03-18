from dataset import MelGANDataset
from models import Generator, Discriminator, Audio2Mel

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import time
import argparse
from utils.config import Config, MelGANConfig

from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--log_dir", default="melgan_logs")
    parser.add_argument("--resume_checkpoint_path", default=None)
    parser.add_argument("--finetune_model_path", default=None)
    parser.add_argument("--data_dir", required=True)

    args = parser.parse_args()
    return args

def save_checkpoint(save_dir, global_step, netG, netD, optimG, optimD):
    save_path = Path(save_dir) / "checkpoint-{}.pth".format(global_step)

    state = {
        "global_step": global_step,
        "net_generator": netG.state_dict(),
        "net_discriminator": netD.state_dict(),
        "optim_generator": optimG.state_dict(),
        "optim_discriminator": optimD.state_dict(),
    }
    print("Saving checkpoint...")
    torch.save(state, save_path)



def train():
    args = parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint_path = Path(args.resume_checkpoint_path) if args.resume_checkpoint_path else None
    finetune_model_path = Path(args.finetune_model_path) if args.finetune_model_path else None
    data_dir = args.data_dir

    # Create logger
    writer = SummaryWriter(str(log_dir))

    # Configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize models
    net_generator = Generator(
        input_size=Config.mel_size,
        ngf=MelGANConfig.ngf,
        n_residual_layers=MelGANConfig.n_residual_layers
    ).to(device=device)

    net_discriminator = Discriminator(
        num_D=MelGANConfig.num_D,
        n_layers=MelGANConfig.n_layers_D,
        n_filters_start=MelGANConfig.ndf,
        down_sampling_factor=MelGANConfig.downsample_factor
    ).to(device)

    fft = Audio2Mel(
        n_mels=Config.mel_size,
        hop_length=256,
        win_length=1024
    ).to(device)

    # Optimizers
    optim_generator = torch.optim.Adam(net_generator.parameters(), lr=MelGANConfig.learning_rate_G, betas=(0.5, 0.9))
    optim_discriminator = torch.optim.Adam(net_discriminator.parameters(), lr=MelGANConfig.learning_rate_D, betas=(0.5, 0.9))
    scaler = torch.amp.GradScaler()

    global_step = 0
    if resume_checkpoint_path and resume_checkpoint_path.exists():
        state = torch.load(resume_checkpoint_path)
        net_generator.load_state_dict(state["net_generator"])
        net_discriminator.load_state_dict(state["net_discriminator"])
        # optim_generator.load_state_dict(state["optim_generator"])
        # optim_discriminator.load_state_dict(state["optim_discriminator"])
        global_step = state['global_step']

    # Create Datasets and Dataloaders
    train_set = MelGANDataset(
        wav_dir=data_dir,
        segment_length=MelGANConfig.seq_len,
    )
    test_set = MelGANDataset(
        wav_dir='C:\\Users\\buryat_saram\\Music\\Project Buryat Saram\\buryat_text_to_speech\\Testing\\wavs2',
        segment_length=4 * Config.sample_rate,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=MelGANConfig.batch_size
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1
    )

    #
    costs = []
    # start = time.time()

    # enable cudnntuner to speedup training

    best_mel_reconst = 1000000
    steps = 0

    for epoch in range(1, MelGANConfig.epochs + 1):
        for iteration, wav in enumerate(train_loader):
            net_discriminator.zero_grad()

            # wav: (bs, 1, n_samples)
            wav = wav.to(device)

            # spec: (bs, n_mels, n_frames)
            spec = fft(wav).detach()

            # wav_pred: (bs, 1, n_samples) : (bs, 1, 8192)
            wav_pred = net_generator(spec)

            with torch.no_grad():
                # spec_pred: (bs, n_mels, n_frames)
                spec_pred = fft(wav_pred.detach())
                S_error = F.l1_loss(spec, spec_pred).item()

            # Train Discriminator
            # spec (bs, n_mels, n_frames)
            # D_fake_det: 4 disc of 7 layers
            # D_real: 4 disc of 7 layers
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                D_fake_det = net_discriminator(wav_pred.detach())
                D_real = net_discriminator(wav)

                loss_D = 0
                for disc_out in D_fake_det:
                    # Compare with the last layer output
                    # disc_out[-1]: (bs, 1, n_samples_reduced)
                    loss_D += F.relu(1 + disc_out[-1]).mean()
                for disc_out in D_real:
                    # Compare with the last layer output
                    # disc_out[-1]: (bs, 1, n_samples_reduced)
                    loss_D += F.relu(1 - disc_out[-1]).mean()

            scaler.scale(loss_D).backward()
            scaler.step(optim_discriminator)
            scaler.update()

            # Train Generator
            net_generator.zero_grad()

            # D_fake: 4 disc of 7 layers
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                D_fake = net_discriminator(wav_pred)

                loss_G = 0
                for disc_out in D_fake:
                    loss_G += -disc_out[-1].mean()

            loss_feat = 0
            feat_weights = 4.0 / (MelGANConfig.n_layers_D + 1)
            D_weights = 1.0 / MelGANConfig.num_D
            wt = D_weights * feat_weights

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                for i in range(MelGANConfig.num_D):
                    for j in range(len(D_fake[i]) - 1):
                        loss_feat += wt * F.l1_loss(D_fake[i][j], D_real[i][j].detach())

            scaler.scale(loss_G + MelGANConfig.lambda_feat * loss_feat).backward()
            scaler.step(optim_generator)
            scaler.update()


            # Update tensorboard
            costs.append([loss_D.item(), loss_G.item(), loss_feat.item(), S_error])
            writer.add_scalar("loss/discriminator", costs[-1][0], global_step)
            writer.add_scalar("loss/generator", costs[-1][1], global_step)
            writer.add_scalar("loss/feature_matching", costs[-1][2], global_step)
            writer.add_scalar("loss/mel_reconstruction", costs[-1][3], global_step)

            if global_step % MelGANConfig.save_interval == 0:
                save_checkpoint(save_dir, global_step, net_generator, net_discriminator, optim_generator, optim_discriminator)

                with torch.no_grad():
                    for j, wav_test_pred in enumerate(test_loader):
                        wav_test_pred = wav_test_pred.to(device)
                        spec = fft(wav_test_pred)
                        pred_audio = net_generator(spec)
                        pred_audio = pred_audio.squeeze().cpu()
                        writer.add_audio(
                            "generated/sample_%d.wav" % j,
                            pred_audio,
                            global_step,
                            sample_rate=Config.sample_rate
                        )

            if global_step % MelGANConfig.log_interval == 0:
                print("Epoch {}, Step {}, Loss={}"
                      .format(epoch, global_step, np.asarray(costs).mean(0)))
                costs = []

            global_step += 1


if __name__ == '__main__':
    train()