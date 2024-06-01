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

    bset_acc = 0.0
    best_epoch = 1
    for epoch in range(args.epochs):
        print("-"*40)
        print(f"Epoch : {epoch + 1}/{args.epochs}")
        epoch_loss = 0.0
        epoch_corrects = 0
        model.train()
        for batch_in, batch_out in tqdm(train_loader):
            batch_in = batch_in.to(device)
            batch_out = batch_out.to(device)

            y_pred = model(batch_in)
            preds = torch.argmax(y_pred,1)

            loss = criterion(y_pred, batch_out)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss = epoch_loss + loss.item() * batch_in.size(0)
            epoch_corrects = epoch_corrects + torch.sum(preds == batch_out.data)
        
        epoch_loss = epoch_loss / len(train_loader.dataset)
        epoch_acc = epoch_corrects.double() / len(train_loader.dataset)

        print(f"train_loss : {epoch_loss:.4f}" acc : {epoch_acc:.4f})
