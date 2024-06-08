import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F

from data_loader.data_loaders import data_loader


def inference(model, test_loader):
    model = torch.load("checkpoints/model.pt") # 미리 학습된 모델을 체크포인트에서 로드
    model.to(device) # 모델을 장치로 이동

    preds = []
    with torch.no_grad(): # 기울기 계산을 비활성화
        model.eval() # 모델을 평가 모드로 전환
        for batch_in, _ in tqdm(test_loader): # 테스트 데이터 로더에서 배치로 데이터 로드
            batch_in = batch_in.to(device)
            y_pred = model(batch_in) # 모델을 통해 예측 수행
            y_pred = torch.argmax(y_pred, 1) # 가장 높은 확률을 가진 클래스 인덱스를 예측값으로 선택
            preds.extend(y_pred.cpu().numpy()) # 예측값을 CPU로 이동시킨 후 리스트에 추가

    submit = pd.read_csv("data/sample_submission.csv")
    submit["Label"] = preds
    submit.to_csv("predicts/predict.csv", index=False) # 예측 결과를 CSV 파일로 저장


if __name__ == "__main__": # 스크립트가 직접 실행될 때만 아래 코드 실행
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("checkpoints/model.pt") # 미리 학습된 모델을 체크포인트에서 로드
    model.to(device) # 모델을 장치로 이동
    inference(model, data_loader("test", batch_size=1024)) # 추론 함수 호출, 테스트 데이터 로더로 예측 수행
