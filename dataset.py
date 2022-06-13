import torch
from torch.utils.data import Dataset

# TODO


class HAM1000Dataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        x = torch.rand((3, 224, 224))
        return x, x, x
