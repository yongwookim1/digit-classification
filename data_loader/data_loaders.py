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

    # 데이터 전처리
    train = np.array(train_data)
    train_size = int(len(train) * 0.9)
    valid_size = len(train) - train_size    

    train_dataset, valid_dataset = random_split(train, (train_size, valid_size))

    train_dataset = np.array(train_dataset)
    valid_dataset = np.array(valid_dataset)
    x_train = train_dataset[:, 1:]
    x_valid = valid_dataset[:, 1:]
    y_train = train_dataset[:, 0]
    y_valid = valid_dataset[:, 0]
    x_test = np.array(test_data)

    x_train = torch.tensor(x_train, dtype=torch.float).view(-1, 1, 28, 28)
    x_valid = torch.tensor(x_valid, dtype=torch.float).view(-1, 1, 28, 28)
    x_test = torch.tensor(x_test, dtype=torch.float).view(-1, 1, 28, 28)

    y_train = torch.tensor(y_train)
    y_valid = torch.tensor(y_valid)
    y_test = torch.randn(len(x_test), 10)  # Dummy labels for test set

    # 이미지 변환 정의
    train_transforms = v2.Compose(
        [
            v2.Resize(32, antialias=True),
            v2.RandomRotation(15),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.1307], std=[0.3081]),
        ]
    )
    test_transforms = v2.Compose(
        [
            v2.Resize(32, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.1307], std=[0.3081]),
        ]
    )

    if type == "train":
        train_dataset = MNIST(x_train, y_train, train_transforms)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=4,
            shuffle=True,
        )
        return train_loader

    if type == "valid":
        valid_dataset = MNIST(x_valid, y_valid, test_transforms)
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            num_workers=4,
            shuffle=False,
        )
        return valid_loader
    if type == "test":
        test_dataset = MNIST(x_test, y_test, test_transforms)
        test_loader = DataLoader(
            test_dataset, batch_size=1, num_workers=4, shuffle=False, drop_last=False
        )
        return test_loader