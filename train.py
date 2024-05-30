import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from models.LeNet5 import LeNet5
from utils.utils import set_seed
from data_loader.data_loaders import data_loader

def train(train_loader, valid_loader, args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet5(num_classes = 10)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr = args.lr, betas = (0.9, 0.999), eps = 1e-08)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)