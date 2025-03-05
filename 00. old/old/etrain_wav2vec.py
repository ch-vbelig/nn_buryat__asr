import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from cmodel import PhoneRecognitionModel, PhoneRecognitionModelWav2Vec
import utils.config as config
import json
from bdataset import SpeechDataset, SpeechDatasetWav2Vec
import dengine as engine
import torch.optim.lr_scheduler as lr_scheduler
from pathlib import Path


def prepare_dataloader(data_path, data_csv, batch_size):
    """
    :param data_path (str): path to *.json file with 'data', 'targets', 'input_lengths', and 'target_lengths' keywords
    :return: train_dataloader (torch.utils.data.DataLoader)
    """
    dataset = SpeechDatasetWav2Vec(data_path, data_csv)
    train_dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
    )
    return train_dataloader


def configure_model(model_path=None, num_classes=config.NUM_OF_PHONE_UNITS, learning_rate=1e-3):
    """
    :param model_path (str): path to an existing model
    :return: model (torch.nn.Module): configured model
    """
    OUTPUT_SIZE = num_classes + 1  # 59 + 1 = 60

    model = PhoneRecognitionModelWav2Vec(
        output_size=OUTPUT_SIZE
    )

    if model_path and Path(model_path).exists():
        print(f"Loading model from {model_path} ...")
        # Load the weights and set to train mode
        model = engine.load_model(model, model_path)
        model.train()
    else:
        print("Configuring asdf new model ...")
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    return model, optimizer


def run_training(model, optimizer, epochs, save_path, audio_path, csv_path, loss_threshold):
    """
    :param model (torch.nn.Module): configured model
    :param optimizer (torch.optim.Adam)
    :param save_path (str): model save path
    :param audio_path: path to *.json file with 'data', 'targets', 'input_lengths', and 'target_lengths' keywords
    """
    dataloader = prepare_dataloader(audio_path, csv_path, config.BATCH_SIZE)

    model.to(config.DEVICE)
    model.freeze_wav2vec()

    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=epochs)

    blank_id = config.NUM_OF_PHONE_UNITS
    criterion = nn.CTCLoss(blank=blank_id)

    for epoch in range(1, epochs + 1):
        loss = engine.train_fn(dataloader, model, optimizer, criterion)

        # if epoch % 5 == 0:
        # loss = engine.eval_fn(test_dataloader, encoder, decoder, criterion)

        print(f"Epoch: {epoch}, loss: {loss}")
        scheduler.step()
        if epoch % 5 == 0 and (loss < loss_threshold):
            loss_threshold = loss
            engine.save_model(model, save_path)
        engine.save_model(model, save_path)


if __name__ == '__main__':
    AUDIO_DIR = '../../c_phone_recognition/data/audio/'
    CSV_DATA_PATH = 'old/bur_phrase_to_phone_wav2vec.csv'
    MODEL_PATH = '../models/phone_model_wav2vec.pth'
    N_CLASSES = config.NUM_OF_PHONE_UNITS
    LR = 1e-3
    EPOCHS = 20
    loss_threshold = 3

    # Configure and train
    model, optimizer = configure_model(model_path=MODEL_PATH, num_classes=N_CLASSES, learning_rate=LR)
    run_training(model, optimizer, EPOCHS, MODEL_PATH, AUDIO_DIR, CSV_DATA_PATH, loss_threshold)
