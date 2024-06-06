import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import v2

import numpy as np


# MNIST 데이터셋 정의 (PyTorch의 Dataset 클래스를 상속)
class MNIST(Dataset):
    def __init__(self, x_data, y_data, transforms):
        # 데이터와 변환 함수 초기화
        self.x_data = x_data
        self.y_data = y_data
        self.transforms = transforms

    def __len__(self):
        # 데이터셋의 전체 샘플 수 반환
        return len(self.x_data)

    def __getitem__(self, idx):
        # 주어진 인덱스에 해당하는 이미지와 라벨 반환
        img = self.x_data[idx]
        transformed_img = self.transforms(img)  # 이미지 변환 적용
        label = self.y_data[idx]
        return transformed_img, label  # 변환된 이미지와 라벨 반환


# 데이터 로드 및 전처리 함수
def data_loader(type, batch_size):
    # 데이터 로드 (CSV 파일에서 데이터프레임으로 불러오기)
    test_data = pd.read_csv("data/test.csv")
    train_data = pd.read_csv("data/train.csv")

    # 데이터 전처리 (넘파이 배열로 변환)
    train = np.array(train_data)
    train_size = int(len(train) * 0.9)  # 훈련 데이터의 90%를 훈련 세트로 사용
    valid_size = len(train) - train_size  # 나머지 10%는 검증 세트로 사용

    # 훈련 데이터와 검증 데이터로 분할
    train_dataset, valid_dataset = random_split(train, (train_size, valid_size))

    # 넘파이 배열로 변환
    train_dataset = np.array(train_dataset)
    valid_dataset = np.array(valid_dataset)
    x_train = train_dataset[:, 1:]  # 훈련 데이터의 이미지 부분
    x_valid = valid_dataset[:, 1:]  # 검증 데이터의 이미지 부분
    y_train = train_dataset[:, 0]  # 훈련 데이터의 라벨 부분
    y_valid = valid_dataset[:, 0]  # 검증 데이터의 라벨 부분
    x_test = np.array(test_data)  # 테스트 데이터

    # 텐서로 변환하고 형상 변경 (28x28 이미지로 재구성)
    x_train = torch.tensor(x_train, dtype=torch.float).view(-1, 1, 28, 28)
    x_valid = torch.tensor(x_valid, dtype=torch.float).view(-1, 1, 28, 28)
    x_test = torch.tensor(x_test, dtype=torch.float).view(-1, 1, 28, 28)

    # 라벨을 텐서로 변환
    y_train = torch.tensor(y_train)
    y_valid = torch.tensor(y_valid)
    y_test = torch.randn(len(x_test), 10)  # 테스트 세트용 더미 라벨 (랜덤)

    # 이미지 변환 정의 (훈련 데이터)
    train_transforms = v2.Compose(
        [
            v2.Resize(32, antialias=True),  # 이미지 크기를 32x32로 조정
            v2.RandomRotation(15),  # 이미지를 무작위로 15도 회전
            v2.ToDtype(torch.float32, scale=True),  # 데이터 타입 변환 및 스케일링
            v2.Normalize(mean=[0.1307], std=[0.3081]),  # 정규화
        ]
    )
    # 이미지 변환 정의 (테스트 데이터)
    test_transforms = v2.Compose(
        [
            v2.Resize(32, antialias=True),  # 이미지 크기를 32x32로 조정
            v2.ToDtype(torch.float32, scale=True),  # 데이터 타입 변환 및 스케일링
            v2.Normalize(mean=[0.1307], std=[0.3081]),  # 정규화
        ]
    )

    # 데이터 로더 타입에 따라 다른 데이터 로더 반환
    if type == "train":
        train_dataset = MNIST(x_train, y_train, train_transforms)  # 훈련 데이터셋
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,  # 배치 크기 설정
            num_workers=4,  # 데이터 로딩에 사용할 워커 수
            shuffle=True,  # 데이터를 무작위로 섞음
        )
        return train_loader

    if type == "valid":
        valid_dataset = MNIST(x_valid, y_valid, test_transforms)  # 검증 데이터셋
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,  # 배치 크기 설정
            num_workers=4,  # 데이터 로딩에 사용할 워커 수
            shuffle=False,  # 데이터를 섞지 않음
        )
        return valid_loader

    if type == "test":
        test_dataset = MNIST(x_test, y_test, test_transforms)  # 테스트 데이터셋
        test_loader = DataLoader(
            test_dataset, 
            batch_size=1,  # 배치 크기 설정 (테스트 데이터셋은 1개씩 로드)
            num_workers=4,  # 데이터 로딩에 사용할 워커 수
            shuffle=False,  # 데이터를 섞지 않음
            drop_last=False  # 마지막 배치를 버리지 않음
        )
        return test_loader
