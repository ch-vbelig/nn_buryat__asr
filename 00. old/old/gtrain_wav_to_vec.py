import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from cmodel import WAV_TO_VEC
import utils.config as config
from bdataset import Wav2VecDataset
import dengine as engine
import torch.optim.lr_scheduler as lr_scheduler
from pathlib import Path
import gc
from torch.nn.utils.rnn import pad_sequence


def pad_batch(batch):
    # waveforms: bs (ts) : bs (ts)
    waveforms = [b[0] for b in batch]

    # targets: n (ts) : n (ts)
    targets = [b[1] for b in batch]

    input_lengths = [b[2] for b in batch]
    target_lengths = [b[3] for b in batch]

    # get padded tensors from lists
    waveforms = pad_sequence(waveforms, padding_value=0)
    waveforms = torch.transpose(waveforms, 0, 1)
    targets = pad_sequence(targets, padding_value=0)

    # targets: (ts, bs) -> (bs, ts)
    targets = targets.transpose(0, 1)

    # get tensors
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    return waveforms, targets, input_lengths, target_lengths


def prepare_dataloader(audio_dir, phone_dir, phone_map_path, batch_size):
    dataset = Wav2VecDataset(audio_dir, phone_dir, phone_map_path)
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=pad_batch,
        shuffle=True
    )

    return train_dataloader, dataset


def configure_model(model_path, num_classes, learning_rate):
    model = WAV_TO_VEC(
        output_size=num_classes
    )

    if model_path and Path(model_path).exists():
        print(f"Loading model from {model_path} ...")
        model = engine.load_model(model, model_path)
        model.train()
    else:
        print("Configuring asdf new model ...")

    print(model)
    # model.freeze_layers()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    return model, optimizer


def run_training(model, optimizer, epochs, save_path, audio_dir, phone_dir, phone_map_path, blank_id):
    """
    :param model (torch.nn.Module): configured model
    :param optimizer (torch.optim.Adam)
    :param save_path (str): model save path
    :param data_path: path to *.json file with 'data', 'targets', 'input_lengths', and 'target_lengths' keywords
    """
    dataloader, dataset = prepare_dataloader(audio_dir, phone_dir, phone_map_path, config.BATCH_SIZE)

    gc.collect()

    model.to(config.DEVICE)
    print(next(model.parameters()).is_cuda)

    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=epochs)

    print('blank id', blank_id)
    criterion = nn.CTCLoss(blank=blank_id)

    loss_threshold = 3

    for epoch in range(1, epochs + 1):
        loss = engine.train_fn(dataloader, model, optimizer, criterion)
        if epoch == 1:
            dataset.save_maps()

        # if epoch % 5 == 0:
        # loss = engine.eval_fn(test_dataloader, encoder, decoder, criterion)

        # if epoch % 5 == 0:
        print(f"Epoch: {epoch}, loss: {loss}")
        scheduler.step()
        if epoch % 5 == 0 and loss < loss_threshold:
            loss_threshold = loss
            engine.save_model(model, save_path)
    engine.save_model(model, save_path)


if __name__ == '__main__':
    AUDIO_DIR = 'E:\Programming\Buryat Speech Recognition\data\\audio'
    PHONE_DIR = 'E:\Programming\Buryat Speech Recognition\data\phones'
    PHONE_MAP_PATH = '../../c_phone_recognition/data/phones_map.json'
    MODEL_PATH = '../../c_phone_recognition/models/phone_model_wav3vec.pth'
    N_CLASSES = 53
    BLANK_ID = 0
    LR = 3e-4
    EPOCHS = 50

    # Configure and train
    model, optimizer = configure_model(
        model_path=MODEL_PATH,
        num_classes=N_CLASSES,
        learning_rate=LR)

    run_training(model, optimizer, EPOCHS, MODEL_PATH, AUDIO_DIR, PHONE_DIR, PHONE_MAP_PATH, BLANK_ID)
