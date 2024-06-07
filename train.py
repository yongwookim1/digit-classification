import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from data_loader.data_loaders import data_loader
from models.LeNet5 import LeNet5
from utils.utils import set_seed


def train(train_loader, valid_loader, args):
    set_seed(args.seed) # 재현성을 위해 시드 설정

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")# cuda사용이 가능하면 사용하도록 설정
    model = LeNet5(num_classes=10)#LeNet5 모델 초기화 및 num_class는 10으로 설정(MNIST 데이터셋 대응)
    model.to(device)

    # Adam 최적화 알고리즘 설정
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08
    )
    # 손실 함수 설정(교차 엔트로피 손실 함수)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    best_acc = 0.0 # 최고 정확도 초기화
    best_epoch = 1 # 최고 정확도가 만들어졌을 때의 에포크 초기화

    #에포크 루프(반복)
    for epoch in range(args.epochs):
        print("-" * 40)
        print(f"Epoch : {epoch+1}/{args.epochs}")

        #학습 루프를 위한 초기화
        epoch_loss = 0.0
        epoch_corrects = 0
        model.train() # 모델 학습 모드 설정

        #train_loader에서 배치 단위로 데이터로 모델 학습
        for batch_in, batch_out in tqdm(train_loader):
            batch_in = batch_in.to(device)
            batch_out = batch_out.to(device)

            #모델 예측
            y_pred = model(batch_in)
            preds = torch.argmax(y_pred, 1)

            # 손실 계산
            loss = criterion(y_pred, batch_out)

            # 역전파 초기화 및 기울기계산을 통한 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_in.size(0)
            epoch_corrects += torch.sum(preds == batch_out.data)

        # 에포크 당 평균 손실과 정확도를 계산 후 출력
        epoch_loss = epoch_loss / len(train_loader.dataset)
        epoch_acc = epoch_corrects / len(train_loader.dataset)
        print(f"Train loss : {epoch_loss:.4f} acc : {epoch_acc:.4f}")

        #검증 루프를 위한 초기화
        epoch_loss = 0.0
        epoch_corrects = 0
        model.eval() # 모델 평가 모드 설정

        # 검증 데이터 로더 루프
        for batch_in, batch_out in tqdm(valid_loader):
            batch_in = batch_in.to(device)
            batch_out = batch_out.to(device)

            with torch.no_grad(): # 검증 때 불필요한 기울기 계산 제외
                y_pred = model(batch_in)
                preds = torch.argmax(y_pred, 1)

            # 손실 계산, 역전파는 사용하지 않음
            epoch_loss += loss.item() * batch_in.size(0)
            epoch_corrects += torch.sum(preds == batch_out.data)

        # 에포크 당 평균 손실과 정확도를 계산
        epoch_loss = epoch_loss / len(valid_loader.dataset)
        epoch_acc = epoch_corrects / len(valid_loader.dataset)

        # 최고 정확도 갱신 시 best_acc, best_epoch 최신화 및 모델 저장
        if epoch_acc >= best_acc:
            best_acc = epoch_acc
            best_epoch = epoch + 1
            torch.save(model, "checkpoints/model.pt")

        # 각 에포크의 검증 손실과 정확도, 최고 정확도와 최고 정확도를 기록한 에포크 출력
        print(f"Valid loss : {epoch_loss:.4f} acc: {epoch_acc:.4f}")
        print(f"Best acc : {best_acc:.4f}")
        print(f"Best epoch : {best_epoch}")
        print("-" * 40)


if __name__ == "__main__":
    # 명령줄 인자 파서 설정
    parser = argparse.ArgumentParser()

    # --seed 명령줄 인자 추가 - 난수 시드 설정(기본값 42)
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    # --epochs 명령줄 인자 추가 - 학습할 에포크 수 설정(기본값 200)
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of epochs to train (default: 200)",
    )
    # --lr (learning_rate) 명령줄 인자 추가 - 학습률 설정(기본값 0.001)
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)"
    )
    # --batch_size 명령줄 인자 추가 - 배치 크기 설정(기본값 1024)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size of train, validation dataset (default: 1024)",
    )

    # 명령줄 인자 파싱
    args = parser.parse_args()
    print(args)

    # 학습 데이터 로더, 검증 데이터 로더 생성 후 학습 함수 호출
    train(
        data_loader("train", batch_size=args.batch_size),
        data_loader("valid", batch_size=args.batch_size),
        args,
    )
