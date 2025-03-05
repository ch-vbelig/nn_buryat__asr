import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from models import Text2Mel, weight_init
from dataset import *
from torch.utils.data import DataLoader
import argparse
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import os
import librosa


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', dest='data_path', required=False, help='Path to the dataset')
parser.add_argument('--text', dest='text_path', required=False, help='Path to texts file')
parser.add_argument('--mel', dest='mel_path', required=False, help='Path to mel spectrograms')
parser.add_argument('-s', '--save', dest='save_dir', required=False, default='checkpoints', help='Where to save checkpoints')
parser.add_argument('-l', '--log', dest='log_dir', required=False, default='logs', help='Where to save logs')
parser.add_argument('-r', '--restore', dest='restore_path', required=False, default=None,
                    help='Checkpoint to continue training from')
parser.add_argument('--batch_size', dest='batch_size', required=False, default=32, type=int)
parser.add_argument('--print_iter', dest='print_iter', required=False, default=10, type=int,
                    help='Print progress every x iterations')
parser.add_argument('--save_iter', dest='save_iter', required=False, default=5000, type=int,
                    help='Save checkpoint every x iterations')
parser.add_argument('--num_workers', dest="num_workers", required=False, default=8,
                    help="Number of processes to use for data loading")
parser.add_argument('--cc', dest="cc", action="store_true",
                    help="Set flag, if you do not want to use the current config file, instead of the config saved with"
                         " the checkpoint.")

def test():
    print(Config.sample_rate)

def plot_matrix(matrix, file):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix.detach().cpu())
    fig.colorbar(im)
    plt.title('{} Steps'.format(global_step))
    plt.savefig(file, format='png')
    plt.close(fig)


if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not args.text_path:
        assert args.data_path is not None, "Data path not given"
        args.text_path = os.path.join(args.data_path, "lines.txt")
    if not args.mel_path:
        assert args.data_path is not None, "Data path not given"
        args.mel_path = os.path.join(args.data_path, "mel")

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    # When loading from checkpoint, first check for the config
    if args.restore_path is not None:
        print("Inspecting checkpoint: {}".format(args.restore_path))
        state = torch.load(args.restore_path, map_location=device)
        conf = state["config"]
        conflicts = False
        warning = "\nWARNING: Saved config does not match with current config file. Conflicts detected:"
        for key, value in conf:
            if getattr(Config, key) != value:
                conflicts = True
                warning += "\n      {}: '{}' vs. '{}'".format(key, value, getattr(Config, key))
        if conflicts:
            print(warning)
            if args.cc:
                print("Will use the current config file.\n")
            else:
                print("Will fall back to saved config. If you want to use the current config file, run with flag "
                      "'-cc'\n")
                Config.set_config(conf)

    # Tensorboard
    writer = SummaryWriter(args.log_dir)

    print("Loading Text2Mel...")
    net = Text2Mel().to(device)
    net.apply(weight_init)

    l1_criterion = nn.L1Loss().to(device)
    bd_criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    global_step = 0

    # Learning rate decay. Noam scheme
    warmup_steps = 4000.0
    def decay(_):
        step = global_step + 1
        return warmup_steps ** 0.5 * min(step * warmup_steps ** -1.5, step ** -0.5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=decay)
    # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-2)

    if args.restore_path is not None:
        print("Restoring from checkpoint: {}".format(args.restore_path))
        global_step = state["global_step"]
        net.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        l1_criterion.load_state_dict(state["l1_criterion"])
        bd_criterion.load_state_dict(state["bd_criterion"])

    print("Loading dataset...")
    dataset = TTSDataset(args.text_path, args.mel_path, None)
    # batch_sampler = BucketBatchSampler(inputs=[d["text"] for d in dataset], batch_size=args.batch_size,
    #                                    bucket_boundaries=[i for i in range(1, Config.max_N - 1, 20)])
    # data_loader = FastDataLoader(dataset,
    #                              # batch_sampler=batch_sampler,
    #                              collate_fn=collate_fn,
    #                              num_workers=args.num_workers
    #                              )

    data_loader = DataLoader(dataset,
                             # batch_sampler=batch_sampler,
                             collate_fn=collate_fn,
                             batch_size=args.batch_size,
                             shuffle=True,
                            # num_workers=args.num_workers
                             )

    def init_w_matrix(max_text_len, max_mel_len):
        # Prepare the W-Matrix for guided attention
        # W (1, max_text_len, max_mel_len)
        W = torch.zeros(1, max_text_len, max_mel_len, dtype=torch.float32)
        for n_pos in range(max_text_len):
            for t_pos in range(max_mel_len):
                W[0, n_pos, t_pos] = 1.0 - np.exp(-(t_pos / float(max_mel_len) - n_pos / float(max_text_len)) ** 2 / (2 * Config.g ** 2))
        return W
    epoch = 0
    print("Start training")
    while global_step < 100000000:
        for i, sample in enumerate(data_loader):
            text = sample["text"].to(device)    # (bs, max_text_len)
            mel = sample["mel"].to(device)  # (bs, max_mel_len, n_mels)

            # Get W matrix
            W = init_w_matrix(max_text_len=text.size(1), max_mel_len=mel.size(1)).to(device)
            W_batched = W.repeat(text.size(0), 1, 1)  # (bs, max_text_len, max_mel_len)

            optimizer.zero_grad()

            # Add zero frame to the input mel
            bs, _, n_mels = mel.size(0), mel.size(1), mel.size(2)
            zero_frame = torch.zeros(bs, 1, n_mels).to(device)
            S = torch.cat((zero_frame, mel[:, :-1, :]), dim=1)  # (bs, ts, n_mels)

            # Run Text2Mel
            # S (transposed): (bs, n_mels, ts)
            # Y_logits: (bs, n_mels, ts)
            # Y: (bs, n_mels, ts)
            # A: (bs, N, T)
            Y_logits, Y, A, _ = net(text, S.transpose(1, 2)) #

            # Y (transposed): (bs, ts, n_mels)
            # mel: (bs, ts, n_mels)
            l1_loss = l1_criterion(Y.transpose(1, 2), mel)  #  mel.transpose(1, 2)
            bd_loss = bd_criterion(Y_logits.transpose(1, 2), mel)  # mel.transpose(1, 2)

            # Loss for guided attention:
            # Pad A matrix with zeros
            # A (bs, max_N, max_T)
            A_pad = F.pad(input=A, pad=[0, W_batched.size(2)-A.size(2), 0, W_batched.size(1)-A.size(1), 0, 0],
                          mode='constant',  value=0)

            att_loss = torch.sum(torch.abs(A_pad * W_batched)) / (A.size(0) * A.size(1) * A.size(2))

            loss = l1_loss + bd_loss + att_loss
            loss.backward()
            # nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            optimizer.step()
            scheduler.step()

            # Tensorboard
            writer.add_scalar('total loss', loss, global_step)
            writer.add_scalar('mel l1 loss', l1_loss, global_step)
            writer.add_scalar('mel bd loss', bd_loss, global_step)
            writer.add_scalar('guided attention loss loss', att_loss, global_step)

            if global_step % args.print_iter == 0:
                print("Epoch {}, Step {}, L1={:.4f}, BD={:.4f}, Att={:.4f}, Total={:.4f}"
                      .format(epoch, global_step, l1_loss, bd_loss, att_loss, loss))

            if global_step % 1000 == 0:
                # Plot attention
                fig, ax = plt.subplots()
                im = ax.imshow(A.detach().cpu()[0])
                fig.colorbar(im)
                plt.title('{} Steps'.format(global_step))
                writer.add_figure('attention matrix', fig, global_step)

            if global_step % args.save_iter == 0:
                state = {
                    "global_step": global_step,
                    "config": Config.get_config(),
                    "model": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "l1_criterion": l1_criterion.state_dict(),
                    "bd_criterion": bd_criterion.state_dict()
                }
                print("Saving checkpoint...")
                torch.save(state, os.path.join(args.save_dir, "checkpoint-{}.pth".format(global_step)))

            global_step += 1
        epoch += 1

