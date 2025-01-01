import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from cmodel import PhoneRecognitionModel
import utils.config as config
import json
from bdataset import SpeechDataset
import dengine as engine
import torch.optim.lr_scheduler as lr_scheduler
from pathlib import Path

EPOCHS = config.EPOCHS
LEARNING_RATE = config.LEARNING_RATE

def load_data(fpath):
    with open(fpath) as fp:
        data = json.load(fp)
        return data


def prepare_dataloader(data_path):
    dataset = SpeechDataset(data_path)
    train_dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
    )
    return train_dataloader


def configure_model(model_path=None):
    OUTPUT_SIZE = 60

    model = PhoneRecognitionModel(
        output_size=OUTPUT_SIZE
    )

    if model_path and Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        optimizer = optim.SGD(model.parameters(), lr=1e-4)

    else:
        optimizer = optim.Adam(model.parameters(), lr=1e-2)

    return model, optimizer


def run_training(model, optimizer, save_path, data_path):

    dataloader = prepare_dataloader(data_path)

    model.to(config.DEVICE)

    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=config.EPOCHS)

    criterion = nn.CTCLoss(blank=59)

    for epoch in range(1, config.EPOCHS + 1):
        loss = engine.train_fn(dataloader, model, optimizer, criterion)

        # if epoch % 5 == 0:
        # loss = engine.eval_fn(test_dataloader, encoder, decoder, criterion)

        # if epoch % 5 == 0:
        print(f"Epoch: {epoch}, loss: {loss}")
        scheduler.step()

    engine.save_model(model, save_path)

if __name__ == '__main__':
    DATA_JSON_PATH = './data/data.json'
    MODEL_PATH = './models/phone_model_normalized2.pth'
    model, optimizer = configure_model(MODEL_PATH)
    run_training(model, optimizer, MODEL_PATH, DATA_JSON_PATH)



