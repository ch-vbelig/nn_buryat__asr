import torch
import torch.nn as nn
import torch.nn.functional as F
from cmodel import PhoneRecognitionModel
import utils.config as config
from tqdm import tqdm

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path, weights_only=True))
    return model


def train_fn(
        data_loader,
        model: PhoneRecognitionModel,
        optimizer,
        ctc_loss
    ):
    model.train()

    total_loss = 0

    for batch_data in tqdm(data_loader, total=len(data_loader)):
        input_tensor, target_tensor, input_lengths, target_lengths = batch_data
        input_tensor = input_tensor.to(config.DEVICE)
        target_tensor = target_tensor.to(config.DEVICE)

        optimizer.zero_grad()

        # outs: ts, bs, output_size
        log_probs = model(input_tensor)

        bs = log_probs.size(1)
        # input_lengths = (input_lengths - 5) // 2

        # input_lengths = torch.full(
        #     size=(bs,),
        #     fill_value=log_probs.size(0),
        #     dtype=torch.int32
        # )
        #
        # target_lengths = torch.full(
        #     size=(bs,),
        #     fill_value=target_tensor.size(1),
        #     dtype=torch.int32
        # )

        loss = ctc_loss(log_probs, target_tensor, input_lengths, target_lengths)
        # calculate gradient and update
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # print(f'batch: {i}, loss: {total_loss}')

    return total_loss / len(data_loader)