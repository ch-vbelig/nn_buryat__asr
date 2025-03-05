import torch
import c_phone_recognition.utils.config as config
from tqdm import tqdm


def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path, weights_only=True))
    return model


def train_fn(
        data_loader,
        model,
        optimizer,
        ctc_loss,
        verbose=False
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


        ts = log_probs.size(0)
        if verbose:
            print('Specs: ', input_tensor.size())
            print('Outputs:', log_probs.size())
            print('Target:', target_tensor.size())
            print()

        # Compute the input lengths after convolution
        input_lengths = torch.floor(input_lengths * (ts / input_tensor.size(-1))).long()
        # print(target_lengths)
        loss = ctc_loss(log_probs, target_tensor, input_lengths, target_lengths)

        # Calculate gradient and update
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)