import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F

from data_loader.data_loaders import data_loader


def inference(model, test_loader):
    model = torch.load("checkpoints/model.pt")
    model.to(device)

    preds = []
    with torch.no_grad():
        model.eval()
        for batch_in, _ in tqdm(test_loader):
            batch_in = batch_in.to(device)
            y_pred = model(batch_in)
            y_pred = torch.argmax(y_pred, 1)
            preds.extend(y_pred.cpu().numpy())

    submit = pd.read_csv("data/sample_submission.csv")
    submit["Label"] = preds
    submit.to_csv("predicts/predict.csv", index=False)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("checkpoints/model.pt")
    model.to(device)
    inference(model, data_loader("test", batch_size=1024))
