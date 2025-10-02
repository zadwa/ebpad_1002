import pandas as pd
import numpy as np
import random
import time
from collections import Counter
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, auc
import torch.nn.functional as F

class MLPEncoder(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=128, encoded_dim=64):
        super(MLPEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, encoded_dim),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = x.to(torch.float32)
        return self.encoder(x)

class MLPDecoder(nn.Module):
    def __init__(self, encoded_dim=64, hidden_dim=128, output_dim=1024):
        super(MLPDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(encoded_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.decoder(x)

class MLPAutoencoder(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=128, encoded_dim=64):
        super(MLPAutoencoder, self).__init__()
        self.encoder = MLPEncoder(input_dim, hidden_dim, encoded_dim)
        self.decoder = MLPDecoder(encoded_dim, hidden_dim, input_dim)

    def forward(self, x):
        x = x.to(torch.float32)
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed

def sampler(X, y, batch_size):
    index_unlabelled = np.where(y == 0)[0]
    index_labelled = np.where(y == 1)[0]
    np.random.seed(42)
    n = 0
    while len(index_unlabelled) >= batch_size:
        index_unlabelled_batch = np.random.choice(index_unlabelled, batch_size//2, replace=False)
        index_unlabelled = np.setdiff1d(index_unlabelled, index_unlabelled_batch)
        index_labelled_batch = np.random.choice(index_labelled, batch_size//2, replace=True)
        index_batch = np.append(index_unlabelled_batch, index_labelled_batch)
        np.random.shuffle(index_batch)
        if n == 0:
            X_new = X[index_batch]
            y_new = y[index_batch]
        else:
            X_new = np.append(X_new, X[index_batch], axis=0)
            y_new = np.append(y_new, y[index_batch])
        if n%50 == 0:
            print("n   ",n)
        n += 1
    train_tensor = TensorDataset(torch.from_numpy(X_new).float(), torch.tensor(y_new))
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader

train_X_all = np.load('')
train_y_all = np.load('')
test_data = np.load('')
test_data = torch.from_numpy(test_data)
test_labels = np.load('')

autoencoder_batch_size = 1024
autoencoder_lr = 0.001
autoencoder_num_epochs = 20
kernel_batch = 256
kernel_lr = 0.00005
kernel_num_epochs = 10
top_k = 500
BND = 0.3

X_unlabelled = train_X_all[train_y_all == 0]
dataset = TensorDataset(torch.from_numpy(X_unlabelled).float())
data_loader = DataLoader(dataset, batch_size=autoencoder_batch_size, shuffle=True)
train_data_loader = sampler(train_X_all, train_y_all, kernel_batch)

auroc_list = []
auprc_list = []
for i in range(5):
    autoencoder_model = MLPAutoencoder().to(torch.float32)
    optimizer = optim.Adam(autoencoder_model.parameters(), lr=autoencoder_lr)
    autoencoder_criterion = nn.MSELoss(reduction='mean')
    autoencoder_model.to('cuda')
    for epoch in range(autoencoder_num_epochs):
        bacth_loss = 0
        for X in data_loader:
            inputs = X[0].to('cuda')
            optimizer.zero_grad()
            outputs = autoencoder_model(inputs)
            loss = autoencoder_criterion(outputs, inputs)
            loss = loss.float()
            loss.backward()
            optimizer.step()
            bacth_loss += loss.item()
        print(f"pretrain:   Epoch {epoch+1}/{autoencoder_num_epochs}      Loss: {bacth_loss/len(data_loader):.4f}")

    with torch.no_grad():
        input_data_unlabelled = torch.from_numpy(X_unlabelled).to('cuda')
        encoded_data = autoencoder_model.encoder(input_data_unlabelled)
        unlabelled_center = encoded_data.mean(dim=0)
        unlabelled_center = unlabelled_center.to('cuda')
        print("unlabelled_center.shape: ", unlabelled_center.shape)
    kernelmodel = autoencoder_model.encoder
    optimizer = optim.Adam(kernelmodel.parameters(), lr=kernel_lr, weight_decay=0.00005)
    for epoch in range(kernel_num_epochs):
        bacth_loss = 0.0
        with torch.no_grad():
            input_data_rep = kernelmodel(input_data_unlabelled)
            input_scores = torch.norm(input_data_rep - unlabelled_center, dim=1)
            _, topk_indices = torch.topk(input_scores, top_k, largest=True)
            topk_embeddings = input_data_rep[topk_indices]
            edge_reference_embedding = topk_embeddings.mean(dim=0)
        distance_unlabelled_centre_edge_reference = torch.norm(edge_reference_embedding - unlabelled_center, dim=0)
        for X, y in train_data_loader:
            optimizer.zero_grad()
            X, y = X.to('cuda'), y.to('cuda')
            encoded_data = kernelmodel(X)
            distances_to_center = torch.norm(encoded_data - unlabelled_center, dim=1)
            distances_to_edge_reference = torch.norm(encoded_data - edge_reference_embedding, dim=1)
            loss = torch.where(
                        y == 0,
                        distances_to_center,
                        torch.where(
                            distances_to_center < distance_unlabelled_centre_edge_reference,
                            torch.maximum(BND - distances_to_center, torch.tensor(0)),
                            torch.maximum(BND - distances_to_center, torch.tensor(0)) + torch.maximum(BND/2 - distances_to_edge_reference, torch.tensor(0))
                        )
                    )
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            bacth_loss += loss.item()
        print(f"train:    [{epoch+1}/{kernel_num_epochs}], Loss: {bacth_loss/len(data_loader):.4f}")
    with torch.no_grad():
        test_data = test_data.to('cuda')
        test_data_rep = kernelmodel(test_data)
        test_scores = torch.norm(test_data_rep - unlabelled_center, dim=1)
        print(test_scores.shape)
        test_scores = test_scores.squeeze().cpu().numpy()
        auroc = roc_auc_score(test_labels, test_scores)
        precision, recall, _ = precision_recall_curve(test_labels, test_scores)
        auprc = auc(recall, precision)
        print(f"AUROC: {auroc:.4f}")
        print(f"AUPRC: {auprc:.4f}")
        auroc_list.append(auroc)
        auprc_list.append(auprc)

result_roc = np.array(auroc_list)
result_prc = np.array(auprc_list)
print("roc:  ")
print("mean:  ",np.mean(auroc_list))
print("std:   ",np.std(result_roc))
print("  ")
print("prc:  ")
print("mean:  ",np.mean(result_prc))
print("std:   ",np.std(result_prc))
print("  ")
print()
print()
print("autoencoder_batch_size    ",autoencoder_batch_size)
print("autoencoder_lr    ",autoencoder_lr)
print("autoencoder_num_epochs    ",autoencoder_num_epochs)
print("kernel_batch    ",kernel_batch)
print("kernel_lr    ",kernel_lr)
print("kernel_num_epochs    ",kernel_num_epochs)
print("top_k:  ",top_k)
print("BND  ",BND)
