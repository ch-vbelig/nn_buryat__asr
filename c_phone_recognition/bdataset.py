import torch
import json
import torch.nn as nn


class SpeechDataset:
    def __init__(self, data_path):
        self.data_path = data_path
        obj = self.load_data()

        self.data = obj['data']
        self.targets = obj['targets']
        self.input_lengths = obj['input_lengths']
        self.target_lengths = obj['target_lengths']

    def load_data(self):
        with open(self.data_path) as fp:
            obj = json.load(fp)
            return obj

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        x = torch.tensor(self.data[item], dtype=torch.float)
        y = torch.tensor(self.targets[item], dtype=torch.long)
        input_lengths = torch.tensor(self.input_lengths[item], dtype=torch.long)
        target_lengths = torch.tensor(self.target_lengths[item], dtype=torch.long)
        return x, y, input_lengths, target_lengths
