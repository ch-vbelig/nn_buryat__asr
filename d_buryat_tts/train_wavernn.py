import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.config import VocoderConfig, Config
from models import WaveRNN
from dataset import WaveRNNDataset

from pathlib import Path


def save_checkpoint(checkpoint_dir, model, optimizer, scheduler, step):

    checkpoint_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step
    }

    checkpoint_path = Path(checkpoint_dir) / f'model_step_{step:09d}.pth'
    torch.save(checkpoint_state, checkpoint_path)
    print(f"Written checkpoint: {checkpoint_path} to disk")


def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    print(f"Loading checkpoint: {checkpoint_path} from disk")

    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model"])
    # optimizer.load_state_dict(checkpoint["optimizer"])
    # scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint["step"]


def train_model(qwav_dir, mel_dir, checkpoint_dir, resume_checkpoint_path=None):

    # specify device to train on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init model
    model = WaveRNN()
    model = model.to(device)
    model.train()

    # Init optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=VocoderConfig.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=VocoderConfig.lr_scheduler_step_size,
        gamma=VocoderConfig.lr_scheduler_gamma
    )
    scaler = torch.amp.GradScaler()

    if resume_checkpoint_path is not None:
        global_step = load_checkpoint(resume_checkpoint_path, model, optimizer, scheduler)
    else:
        global_step = 0

    # Init the DataLoader
    dataset = WaveRNNDataset(qwav_dir=qwav_dir, mel_dir=mel_dir, max_items=None)
    dataloader = DataLoader(
        dataset,
        batch_size=VocoderConfig.batch_size,
        shuffle=True,
    )

    num_epochs = VocoderConfig.num_steps // len(dataloader) + 1
    start_epoch = global_step // len(dataloader) + 1

    criterion = nn.CrossEntropyLoss()
    print_interval = 10
    threshold = 0.1

    for epoch in range(start_epoch, num_epochs + 1):
        avg_loss = 0

        for idx, (mels, qwavs) in enumerate(dataloader, 1):

            mels, qwavs = mels.to(device), qwavs.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                # wav_hat: (bs, ts, 1024)
                wav_hat = model(mels, qwavs[:, :-1], normalize=False)
                wav_hat = wav_hat.transpose(1, 2)   # (bs, 1024, ts)
                loss = criterion(wav_hat, qwavs[:, 1:])

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            avg_loss += loss.item()

            if global_step % VocoderConfig.checkpoint_interval == 0:
                save_checkpoint(checkpoint_dir, model, optimizer, scheduler, global_step)

            if global_step % print_interval == 0:
                if (avg_loss / print_interval) < threshold:
                    save_checkpoint(checkpoint_dir, model, optimizer, scheduler, global_step)
                print(f"Epoch {epoch}, Step {global_step} / {VocoderConfig.num_steps}, Loss: {avg_loss / print_interval:.6f}, Current LR: {scheduler.get_last_lr()}")
                avg_loss = 0



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the WaveRNN model")

    parser.add_argument(
        "--data_mel_dir",
        help="Path to dir containing the mels",
        required=True
    )

    parser.add_argument(
        "--data_qwav_dir",
        help="Path to dir containing the qwavs",
        required=True
    )

    parser.add_argument(
        "--checkpoint_dir",
        help="Path to dir to save checkpoints",
        default='wavernn_checkpoints'
    )

    parser.add_argument(
        "--resume_checkpoint_path",
        help="If specified, load checkpoints and resume training from that point",
    )

    args = parser.parse_args()

    mel_dir = args.data_mel_dir
    qwav_dir = args.data_qwav_dir
    checkpoint_dir = args.checkpoint_dir

    resume_checkpoint_path = args.resume_checkpoint_path

    train_model(qwav_dir, mel_dir, checkpoint_dir, resume_checkpoint_path)