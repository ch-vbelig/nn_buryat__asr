import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from cmodel import PhoneRecognitionModel, PhoneRecognitionModelResidual
import utils.config as config
import json
from bdataset import SpeechDataset
import dengine as engine
import torch.optim.lr_scheduler as lr_scheduler
from pathlib import Path
import gc

def prepare_dataloader(data_path, batch_size):
    """
    :param data_path (str): path to *.json file with 'data', 'targets', 'input_lengths', and 'target_lengths' keywords
    :return: train_dataloader (torch.utils.data.DataLoader)
    """
    dataset = SpeechDataset(data_path)
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    return train_dataloader


def configure_model(model_path=None, num_classes=config.N_CLASSES, learning_rate=1e-3):
    """
    :param model_path (str): path to an existing model
    :return: model (torch.nn.Module): configured model
    """
    OUTPUT_SIZE = num_classes + 1  # 59 + 1 = 60

    model = PhoneRecognitionModelResidual(
        output_size=OUTPUT_SIZE
    )

    if model_path and Path(model_path).exists():
        print(f"Loading model from {model_path} ...")
        # Load the weights and set to train mode
        model = engine.load_model(model, model_path)
        model.train()
    else:
        print("Configuring a new model ...")
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    return model, optimizer


def run_training(model, optimizer, epochs, save_path, data_path):
    """
    :param model (torch.nn.Module): configured model
    :param optimizer (torch.optim.Adam)
    :param save_path (str): model save path
    :param data_path: path to *.json file with 'data', 'targets', 'input_lengths', and 'target_lengths' keywords
    """
    dataloader = prepare_dataloader(data_path, config.BATCH_SIZE)

    gc.collect()

    model.to(config.DEVICE)
    print(next(model.parameters()).is_cuda)

    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=epochs)

    blank_id = config.N_CLASSES 
    criterion = nn.CTCLoss(blank=blank_id)

    loss_threshold = 3

    for epoch in range(1, epochs + 1):
        loss = engine.train_fn(dataloader, model, optimizer, criterion)

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
    DATA_JSON_PATH = './data/data_2.json'
    MODEL_PATH = 'models/phone_model_conv_residual.pth'
    N_CLASSES = config.N_CLASSES
    LR = 1e-5
    EPOCHS = config.EPOCHS

    # Configure and train
    model, optimizer = configure_model(MODEL_PATH, N_CLASSES, LR)
    run_training(model, optimizer, EPOCHS, MODEL_PATH, DATA_JSON_PATH)
