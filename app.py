import streamlit as st
import io
import numpy as np
from PIL import Image

import torch
from torchvision.transforms import v2

st.set_page_config(layout="wide")  # 페이지 레이아웃을 wide로 설정


# 이미지를 변환하는 함수
def transform_image(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes))  # 바이트 데이터를 이미지로 변환
    image = image.convert("L")  # 이미지를 그레이스케일로 변환
    image = np.array(image)  # 이미지를 numpy 배열로 변환
    image = torch.tensor(image)  # numpy 배열을 텐서로 변환
    image = image.view(
        -1, 1, 28, 28
    )  # 텐서 모양을 (배치 크기, 채널, 높이, 너비)로 변경

    test_transforms = v2.Compose(
        [
            v2.Resize(32, antialias=True),  # 이미지를 32x32로 크기 조정
            v2.ToDtype(
                torch.float32, scale=True
            ),  # 데이터 타입을 float32로 변환하고 정규화
            v2.Normalize(
                mean=[0.1307], std=[0.3081]
            ),  # 평균과 표준편차를 사용하여 정규화
        ]
    )
    transformed_img = test_transforms(image)  # 변환 적용
    return transformed_img

# 메인 함수
def main():
    st.title("Digit Classification Model")  # 앱 제목 설정

    model = torch.load("checkpoints/model.pt")  # 모델 불러오기
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # CUDA를 사용할 수 있는지 확인
    model.to(device)  # 모델을 장치로 이동
    model.eval()  # 모델 평가 모드 설정


if __name__ == "__main__":
    main()  # 메인 함수 실행

