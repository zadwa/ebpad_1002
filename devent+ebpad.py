import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
from tqdm import tqdm
from sklearn.metrics import auc, roc_curve, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader, Dataset, TensorDataset
import random


class mlp(nn.Module):
    def __init__(self, rep_dim=64):
        super().__init__()
        self.rep_dim = rep_dim
        self.fc1 = nn.Linear(1024, 100, bias=False)
        self.activation_1 = nn.ReLU()
        self.fc2 = nn.Linear(100, self.rep_dim)
        self.activation_2 = nn.ReLU()
        self.score_fc1 = nn.Linear(self.rep_dim, 20, bias=False)
        self.score_fc2 = nn.Linear(20, 1, bias=False)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.fc1(x)
        x = self.activation_1(x)
        x = self.fc2(x)
        x = self.activation_2(x)
        x = self.score_fc1(x)
        x = self.score_fc2(x)
        return x.squeeze()


class MyDataset(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.labels = y.astype(int)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y


class DeviationLoss(nn.Module):
    def __init__(self, confidence_margin=5):
        super().__init__()
        self.confidence_margin = confidence_margin

    def forward(self, y_pred, topk_anomaly_mean, y_true):
        confidence_margin = self.confidence_margin
        ref = torch.normal(mean=0., std=torch.ones(5000)).to('cuda')
        ref_1 = torch.normal(mean=topk_anomaly_mean, std=torch.ones(5000)).to('cuda')
        dev = (y_pred - torch.mean(ref)) / torch.std(ref)
        dev_1 = (y_pred - torch.mean(ref_1)) / torch.std(ref_1)
        normal_loss = torch.abs(dev)
        anomaly_loss = torch.abs((confidence_margin - dev).clamp_(min=0.))
        out_edge_loss = torch.abs((confidence_margin - dev_1).clamp_(min=0.))
        return torch.mean((1 - y_true.float()) * normal_loss + y_true.float() * (5 * anomaly_loss + 1 * out_edge_loss))


class Trainer(object):
    def __init__(self, model, x_train, train_loader, test_set, lr=1e-3, weight_decay=1e-5, confidence_margin=5):
        self.model = model
        self.x_train = x_train
        self.train_loader = train_loader
        self.test_set = test_set
        self.criterion = DeviationLoss(confidence_margin=confidence_margin)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.model = self.model.to('cuda')
        self.criterion = self.criterion.to('cuda')

    def train(self, epochs):
        for epoch in range(epochs):
            train_loss = 0.0
            with torch.no_grad():
                anomaly_output = model(torch.from_numpy(self.x_train).float().to('cuda')).squeeze()
                topk_anomaly, topk_indices = torch.topk(anomaly_output, 500, largest=True)
                topk_anomaly_mean = topk_anomaly.mean().cpu()
            self.model.train()
            for i, (input, target) in enumerate(self.train_loader):
                input, target = input.float(), target.float()
                input, target = input.to('cuda'), target.to('cuda')
                output = self.model(input)
                loss = self.criterion(output, topk_anomaly_mean, target.unsqueeze(1))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            print(f"epoch_number = {epoch + 1}/{epochs} ,   loss = {train_loss / len(train_loader):.4f}")

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            test_data = self.test_set.data
            test_labels = self.test_set.labels
            test_data = torch.from_numpy(test_data).float().to('cuda')
            test_scores = self.model(test_data)
            print(test_scores.shape)
            test_scores = test_scores.squeeze().cpu().numpy()
            auroc = roc_auc_score(test_labels, test_scores)
            precision, recall, _ = precision_recall_curve(test_labels, test_scores)
            auprc = auc(recall, precision)
            print(f"AUROC: {auroc:.4f}")
            print(f"AUPRC: {auprc:.4f}")
            return auroc, auprc


if __name__ == '__main__':
    batch_size = 128
    lr = 0.0001
    weight_decay = 0.00005
    epochs = 20
    confidence_margin = 0.3

    train_x = np.load('')
    train_2_y = np.load('')
    train_x_unlabelled = train_x[train_2_y == 0]
    test_x = np.load('')
    test_2_y = np.load('')

    train_tensor = TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_2_y).float())
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=False, drop_last=True)
    test_set = MyDataset(test_x, test_2_y)

    result_roc = []
    result_prc = []
    for i in range(5):
        model = mlp()
        trainer = Trainer(model, train_x_unlabelled, train_loader, test_set, lr=lr, weight_decay=weight_decay,
                          confidence_margin=confidence_margin)
        trainer.train(epochs)
        auroc, auprc = trainer.eval()
        result_roc.append(auroc)
        result_prc.append(auprc)

    print("roc:  ")
    print("mean:  ", np.mean(result_roc))
    print("std:   ", np.std(result_roc))
    print("  ")
    print("prc:  ")
    print("mean:  ", np.mean(result_prc))
    print("std:   ", np.std(result_prc))
    print("  ")
    print("batch_size:   ", batch_size)
    print("lr:  ", lr)
    print("weight_decay:  ", weight_decay)
    print("epochs:   ", epochs)
    print("condidence_margin:   ", confidence_margin)
