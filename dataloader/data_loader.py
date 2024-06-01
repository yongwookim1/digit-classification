import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import v2

import numpy as np

# MNIST Dataset 정의
class MNIST(Dataset):
    def __init__(self, x_data, y_data, transforms):
        self.x_data = x_data
        self.y_data = y_data
        self.transforms = transforms

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        img = self.x_data[idx]
        transformed_img = self.transforms(img)
        label = self.y_data[idx]
        return transformed_img, label
