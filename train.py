import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from data_loader.data_loaders import data_loader
from models.LeNet5 import LeNet5
from utils.utils import set_seed


def train(train_loader, valid_loader, args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet5(num_classes=10)
    model.to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08
    )
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    best_acc = 0.0
    best_epoch = 1
    for epoch in range(args.epochs):
        print("-" * 40)
        print(f"Epoch : {epoch+1}/{args.epochs}")
        epoch_loss = 0.0
        epoch_corrects = 0
        model.train()
        for batch_in, batch_out in tqdm(train_loader):
            batch_in = batch_in.to(device)
            batch_out = batch_out.to(device)

            y_pred = model(batch_in)
            preds = torch.argmax(y_pred, 1)

            loss = criterion(y_pred, batch_out)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_in.size(0)
            epoch_corrects += torch.sum(preds == batch_out.data)

        epoch_loss = epoch_loss / len(train_loader.dataset)
        epoch_acc = epoch_corrects / len(train_loader.dataset)

        print(f"Train loss : {epoch_loss:.4f} acc : {epoch_acc:.4f}")

        epoch_loss = 0.0
        epoch_corrects = 0
        model.eval()
        for batch_in, batch_out in tqdm(valid_loader):
            batch_in = batch_in.to(device)
            batch_out = batch_out.to(device)

            with torch.no_grad():
                y_pred = model(batch_in)
                preds = torch.argmax(y_pred, 1)

            epoch_loss += loss.item() * batch_in.size(0)
            epoch_corrects += torch.sum(preds == batch_out.data)

        epoch_loss = epoch_loss / len(valid_loader.dataset)
        epoch_acc = epoch_corrects / len(valid_loader.dataset)

        if epoch_acc >= best_acc:
            best_acc = epoch_acc
            best_epoch = epoch + 1
            torch.save(model, "checkpoints/model.pt")

        print(f"Valid loss : {epoch_loss:.4f} acc: {epoch_acc:.4f}")
        print(f"Best acc : {best_acc:.4f}")
        print(f"Best epoch : {best_epoch}")
        print("-" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of epochs to train (default: 200)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size of train, validation dataset (default: 1024)",
    )

    args = parser.parse_args()
    print(args)

    train(
        data_loader("train", batch_size=args.batch_size),
        data_loader("valid", batch_size=args.batch_size),
        args,
    )