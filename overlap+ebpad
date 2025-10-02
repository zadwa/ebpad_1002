import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import nn
from torch.distributions.distribution import Distribution
from torch.distributions import MultivariateNormal, Normal
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix, roc_curve
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys
from collections import Counter

train_X_all = np.load('')
train_y_all = np.load('')
test_data = np.load('')
test_data = torch.from_numpy(test_data)
test_labels = np.load('')

def sampler(X_train, train_3_labels, batch_size):
    index_unlabelled = np.where(train_3_labels == 0)[0]
    index_labelled = np.where(train_3_labels == 1)[0]
    index_edge_instance = np.where(train_3_labels == -100)[0]
    np.random.seed(42)
    n = 0
    while len(index_unlabelled) >= batch_size:
        index_unlabelled_batch = np.random.choice(index_unlabelled, batch_size//2, replace=False)
        index_unlabelled = np.setdiff1d(index_unlabelled, index_unlabelled_batch)
        index_labelled_batch = np.random.choice(index_labelled, batch_size//4, replace=True)
        index_edge_instance_batch = np.random.choice(index_edge_instance, batch_size//4, replace=True)
        index_batch = np.append(index_unlabelled_batch, index_labelled_batch)
        index_batch = np.append(index_batch, index_edge_instance_batch)
        np.random.shuffle(index_batch)
        if n == 0:
            train_new_features = X_train[index_batch]
            train_new_3_labels = train_3_labels[index_batch]
        else:
            train_new_features = np.append(train_new_features, X_train[index_batch], axis=0)
            train_new_3_labels = np.append(train_new_3_labels, train_3_labels[index_batch])
        n += 1
        train_tensor = TensorDataset(torch.from_numpy(train_new_features).float(), torch.tensor(train_new_3_labels))
        train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader

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
        self.batch_norm = nn.BatchNorm1d(num_features=1)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.fc1(x)
        x = self.activation_1(x)
        x = self.fc2(x)
        x = self.activation_2(x)
        x = self.score_fc1(x)
        x = self.score_fc2(x)
        x = self.batch_norm(x)
        return x.squeeze()

class GaussianKDE(Distribution):
    def __init__(self, X, bw, lam=1e-4, device=None):
        self.X = X
        self.bw = bw
        self.dims = X.shape[-1]
        self.n = X.shape[0]
        self.mvn = MultivariateNormal(loc=torch.zeros(self.dims).to(device), covariance_matrix=torch.eye(self.dims).to(device))
        self.lam = lam

    def sample(self, num_samples):
        idxs = (np.random.uniform(0, 1, num_samples) * self.n).astype(int)
        norm = Normal(loc=self.X[idxs], scale=self.bw)
        return norm.sample()

    def score_samples(self, Y, X=None):
        if X == None:
            X = self.X
        log_probs = torch.log((self.bw ** (-self.dims) * torch.exp(self.mvn.log_prob((X.unsqueeze(1) - Y) / self.bw))).sum(dim=0) / self.n + self.lam)
        return log_probs

    def log_prob(self, Y):
        X_chunks = self.X.split(1000)
        Y_chunks = Y.split(1000)
        log_prob = 0
        for x in X_chunks:
            for y in Y_chunks:
                log_prob += self.score_samples(y, x).sum(dim=0)
        return log_prob

def overlap_loss(score_unlabelled, score_labelled, score_edge_instance):
    score_unlabelled = score_unlabelled.reshape(-1, 1)
    score_labelled = score_labelled.reshape(-1, 1)
    score_edge_instance = score_edge_instance.reshape(-1, 1)
    kde_unlabelled = GaussianKDE(X=score_unlabelled, bw=1, device='cuda')
    kde_labelled = GaussianKDE(X=score_labelled, bw=1, device='cuda')
    kde_edge_instance = GaussianKDE(X=score_edge_instance, bw=1, device='cuda')
    xmin = torch.min(torch.min(score_unlabelled), torch.min(score_labelled))
    xmin = torch.min(xmin, torch.min(score_edge_instance))
    xmax = torch.max(torch.max(score_unlabelled), torch.max(score_labelled))
    xmax = torch.max(xmax, torch.max(score_edge_instance))
    dx = 0.2 * (xmax - xmin)
    xmin -= dx
    xmax += dx
    x = torch.linspace(xmin.detach(), xmax.detach(), 1000).to('cuda')
    kde_unlabelled_x = torch.exp(kde_unlabelled.score_samples(x.reshape(-1, 1)))
    kde_labelled_x = torch.exp(kde_labelled.score_samples(x.reshape(-1, 1)))
    kde_edge_instance_x = torch.exp(kde_edge_instance.score_samples(x.reshape(-1, 1)))
    intersection_points_idx_unlabelled_labelled = torch.where(torch.diff(torch.sign(kde_unlabelled_x - kde_labelled_x)))[0].cpu()
    intersection_points_idx_edge_instance_labelled = torch.where(torch.diff(torch.sign(kde_edge_instance_x - kde_labelled_x)))[0].cpu()
    area = 0
    if intersection_points_idx_unlabelled_labelled.size(0) >= 1:
        c = x[np.random.choice(intersection_points_idx_unlabelled_labelled.numpy(), 1)]
        idx_unlabelled = torch.where(x > c)[0]
        idx_labelled = torch.where(x < c)[0]
        area_unlabelled = torch.trapz(kde_unlabelled_x[idx_unlabelled], x[idx_unlabelled])
        area_labelled = torch.trapz(kde_labelled_x[idx_labelled], x[idx_labelled])
        area = area + weight_unlabelled_target * (area_unlabelled + area_labelled)
    else:
        area_unlabelled = torch.trapz(kde_unlabelled_x, x)
        area_labelled = torch.trapz(kde_labelled_x, x)
        area_unlabelled_labelled = (area_unlabelled + area_labelled) / 2
        area = area + area_unlabelled_labelled
    if intersection_points_idx_edge_instance_labelled.size(0) >= 1:
        c = x[np.random.choice(intersection_points_idx_edge_instance_labelled.numpy(), 1)]
        idx_edge_instance = torch.where(x > c)[0]
        idx_labelled = torch.where(x < c)[0]
        area_edge_instance = torch.trapz(kde_edge_instance_x[idx_edge_instance], x[idx_edge_instance])
        area_labelled = torch.trapz(kde_labelled_x[idx_labelled], x[idx_labelled])
        area = area + weight_nontarget_target * (area_edge_instance + area_labelled)
    else:
        area_edge_instance = torch.trapz(kde_edge_instance_x, x)
        area_labelled = torch.trapz(kde_labelled_x, x)
        area = area + (area_edge_instance + area_labelled) / 2
    return area

def train(train_x, train_y, model, epoch):
    for epoch_number in range(epoch):
        model.to('cpu')
        train_x_copy = train_x.copy()
        train_x_copy = torch.from_numpy(train_x_copy).float()
        train_y_copy = train_y.copy()
        model.eval()
        with torch.no_grad():
            scores = model(train_x_copy)
        scores = scores.numpy()
        zero_labels_indices = np.where(train_y_copy == 0)[0]
        zero_labels_scores = scores[zero_labels_indices]
        top_indices = np.argsort(zero_labels_scores)[-int(top_k * len(zero_labels_scores)):]
        real_indices = zero_labels_indices[top_indices]
        train_y_copy[real_indices] = -100
        train_loader = sampler(train_x, train_y_copy, batch_size)
        model.train()
        loss_batch = []
        model.to('cuda')
        for i, data in enumerate(train_loader):
            trainloader_data_features, trainloader_3_labels = data
            trainloader_data_features = trainloader_data_features.to('cuda')
            trainloader_3_labels = trainloader_3_labels.to('cuda')
            model.zero_grad()
            score = model(trainloader_data_features)
            score_unlabelled = score[torch.where(trainloader_3_labels == 0)[0]]
            score_labelled = score[torch.where(trainloader_3_labels == 1)[0]]
            score_edge_instance = score[torch.where(trainloader_3_labels == -100)[0]]
            loss = overlap_loss(score_unlabelled, score_labelled, score_edge_instance)
            loss.backward()
            optimizer.step()
            loss_batch.append(loss.item())
        print("epoch_number =", epoch_number+1, "/", epoch, "   loss =", np.mean(loss_batch))

def test(model, test_data, test_labels):
    with torch.no_grad():
        model.to('cuda')
        model.eval()
        test_data = test_data.to('cuda')
        scores_test = model(test_data)
        scores_test = scores_test.cpu().detach().numpy().tolist()
        test_real_labels = test_labels.tolist()
        test_roc = roc_auc_score(test_real_labels, scores_test)
        precision, recall, _ = precision_recall_curve(test_real_labels, scores_test)
        test_prc = auc(recall, precision)
        print("roc:   ", test_roc, "       prc:  ", test_prc)
        print("  ")
        return test_roc, test_prc

batch_size = 512
learning_rate = 0.00001
epoch_number = 7
weight_unlabelled_target = 1
weight_nontarget_target = 1
top_k = 0.1

print()
print()
print()
print("para：   ")
print("learning_rate:   ", learning_rate)
print("batch_size:  ", batch_size)
print("epoch_number:    ", epoch_number)
print("weight_unlabelled_target:    ", weight_unlabelled_target)
print("weight_nontarget_target:    ", weight_nontarget_target)
print("top_k:   ", top_k)
print()
print()
print()

result_roc = []
result_prc = []

for i in range(5):
    model = mlp(64)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.00005)
    train(train_X_all, train_y_all, model=model, epoch=epoch_number)
    test_roc, test_prc = test(model, test_data, test_labels)
    result_roc.append(test_roc)
    result_prc.append(test_prc)

result_roc = np.array(result_roc)
result_prc = np.array(result_prc)
print("roc:  ")
print("mean:  ", np.mean(result_roc))
print("std:   ", np.std(result_roc))
print("  ")
print("prc:  ")
print("mean:  ", np.mean(result_prc))
print("std:   ", np.std(result_prc))
print("  ")

print()
print()
print()
print("para：   ")
print("learning_rate:   ", learning_rate)
print("batch_size:  ", batch_size)
print("epoch_number:    ", epoch_number)
print("weight_unlabelled_target:    ", weight_unlabelled_target)
print("weight_nontarget_target:    ", weight_nontarget_target)
print("top_k:   ", top_k)
print()
print()
print()
