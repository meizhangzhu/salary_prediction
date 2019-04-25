import code
import json
import random

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataSource(object):
    def __init__(self, dataset, norm_stats, y_scaler=None, standardize_x=False, standardize_y=False):
        self.dataset = []
        self.cur_idx = 0
        self.norm_stats = norm_stats
        self.y_scaler = None

        with open("../data/dataset.json") as f:
            dataset_json = json.load(f)

        if dataset == "train":
            self.dataset = dataset_json["train"]
        elif dataset == "dev":
            self.dataset = dataset_json["dev"]
        elif dataset == "test":
            self.dataset = dataset_json["test"]
        # self.dataset = self.dataset[:100]

        if standardize_x:
            X, Y = zip(*self.dataset)
            X, Y = np.array(X).astype("float"), np.array(Y).astype("float")
            print("normalizing data...")
            # yearsExperience
            X[:, -2] -= self.norm_stats["yearsExperience"]["min"]
            X[:, -2] /= (self.norm_stats["yearsExperience"]["max"] - self.norm_stats["yearsExperience"]["min"])
            # milesFromMetropolis
            X[:, -1] -= self.norm_stats["milesFromMetropolis"]["min"]
            X[:, -1] /= (self.norm_stats["milesFromMetropolis"]["max"] - self.norm_stats["milesFromMetropolis"]["min"])
            X = X.tolist()
            Y = Y.tolist()
            self.dataset = list(zip(X, Y))

        if standardize_y:
            X, Y = zip(*self.dataset)
            X, Y = np.array(X).astype("float"), np.array(Y).astype("float")
            if y_scaler is None:
                self.y_scaler = StandardScaler()
                self.y_scaler.fit(Y.reshape(-1, 1))
            else:
                self.y_scaler = y_scaler
            Y = self.y_scaler.transform(Y.reshape(-1, 1)).reshape(-1)
            X = X.tolist()
            Y = Y.tolist()
            self.dataset = list(zip(X, Y))

    def epoch_init(self, shuffle=False):
        if shuffle:
            random.shuffle(self.dataset)
        self.cur_idx = 0

    def next(self, batch_size):
        if self.cur_idx == len(self.dataset):
            return None

        from_idx = self.cur_idx
        to_idx = min(from_idx+batch_size, len(self.dataset))
        self.cur_idx = to_idx

        batch_data = self.dataset[from_idx:to_idx]
        X, Y = zip(*batch_data)

        X = torch.FloatTensor(X).to(DEVICE)
        Y = torch.FloatTensor(Y).to(DEVICE)

        X_disc = X[:, :5].contiguous().long()
        X_cont = X[:, 5:].contiguous().float()

        # code.interact(local=locals())

        return {
            "X_disc": X_disc,
            "X_cont": X_cont,
            "Y": Y,
        }


