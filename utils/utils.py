import numpy as np
import random

import torch


def set_seed(seed):
    # PyTorch에서 무작위 수 생성을 위한 시드 설정
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CUDA 연산에서 결정론적 동작을 보장하기 위한 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed) # numpy의 무작위 수 생성을 위한 시드 설정
    random.seed(seed) # Python 내장 random 라이브러리의 무작위 수 생성을 위한 시드 설정