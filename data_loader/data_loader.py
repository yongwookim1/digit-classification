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

# 데이터 로드 및 전처리 함수
def data_loader(type, batch_size):
    # 데이터 로드
    test_data = pd.read_csv("data/test.csv")
    train_data = pd.read_csv("data/train.csv")