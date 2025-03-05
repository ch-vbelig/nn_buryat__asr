import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from cmodel import PhoneRecognitionModelResidual
import utils.config as config
from bdataset import SpeechDataset
import dengine as engine
import torch.optim.lr_scheduler as lr_scheduler
from pathlib import Path
import gc
from torch.nn.utils.rnn import pad_sequence


def pad_batch(batch):
    # specs: bs (ts, n_channels, n_mels) : bs (ts, 1, 128)
    specs = [b[0] for b in batch]

    # targets: n (ts) : n (ts)
    targets = [b[1] for b in batch]

    input_lengths = [b[2] for b in batch]
    target_lengths = [b[3] for b in batch]

    # get padded tensors from lists
    specs = pad_sequence(specs, padding_value=-1)
    targets = pad_sequence(targets, padding_value=0)

    # specs: (ts, bs, n_channels, n_mels) -> (bs, 1, 128, ts)
    specs = specs.permute(1, 2, 3, 0)

    # targets: (ts, bs) -> (bs, ts)
    targets = targets.transpose(0, 1)

    # get tensors
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    return specs, targets, input_lengths, target_lengths


def prepare_dataloader(audio_dir, phone_dir, phone_map_path, batch_size):
    dataset = SpeechDataset(audio_dir, phone_dir, phone_map_path)
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=pad_batch,
        shuffle=True
    )

    return train_dataloader, dataset


def configure_model(model_path, num_classes, learning_rate):
    model = PhoneRecognitionModelResidual(
        output_size=num_classes
    )

    if model_path and Path(model_path).exists():
        print(f"Loading model from {model_path} ...")
        model = engine.load_model(model, model_path)
        model.train()
    else:
        print("Configuring asdf new model ...")

    print(model)
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

    loss_threshold = 0.33

    for epoch in range(1, epochs + 1):
        loss = engine.train_fn(dataloader, model, optimizer, criterion, verbose=False)

        print(f"Epoch: {epoch}, loss: {loss}")
        scheduler.step()
        if epoch % 5 == 0 and loss < loss_threshold:
            loss_threshold = loss
            engine.save_model(model, save_path)
    engine.save_model(model, save_path)


if __name__ == '__main__':
    AUDIO_DIR = 'E:\\Programming\\Buryat Speech Recognition\\data\\audio'
    PHONE_DIR = 'E:\\Programming\\Buryat Speech Recognition\\data\\phones'
    # AUDIO_DIR = './data/audio'
    # PHONE_DIR = './data/phones'
    PHONE_MAP_PATH = './data/phones_map.json'
    MODEL_PATH = 'models/phone_model_conv_residual.pth'
    N_CLASSES = config.NUM_OF_PHONE_UNITS + 1
    BLANK_ID = N_CLASSES - 1
    LR = 1e-8
    EPOCHS = 100

    # Configure and train
    model, optimizer = configure_model(
        model_path=MODEL_PATH,
        num_classes=N_CLASSES,
        learning_rate=LR)

    run_training(model, optimizer, EPOCHS, MODEL_PATH, AUDIO_DIR, PHONE_DIR, PHONE_MAP_PATH, BLANK_ID)
