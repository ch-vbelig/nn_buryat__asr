import torch
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
    # Set the model to train mode
    model.train()

    total_loss = 0

    for batch_data in tqdm(data_loader, total=len(data_loader)):

        input_tensor, target_tensor, input_lengths, target_lengths = batch_data
        input_tensor = input_tensor.to(config.DEVICE)
        target_tensor = target_tensor.to(config.DEVICE)

        # Zero the gradients
        optimizer.zero_grad()

        # log_probs: ts, bs, output_size
        log_probs = model(input_tensor)
        # log_probs = log_probs.permute(1, 0, 2)

        ts = log_probs.size(0)

        # Compute the input lengths after convolution
        input_lengths = torch.ceil(input_lengths * (ts / input_tensor.size(-1))).long()

        loss = ctc_loss(log_probs, target_tensor, input_lengths, target_lengths)

        # Calculate gradient and update
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)