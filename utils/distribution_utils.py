import logging
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from copy import deepcopy




def train_distribution_diversity(n_distribution, n_dim, max_iters=1000):

    n_mean = nn.Parameter(torch.randn(n_distribution, n_dim))
    temp_matrix = 1 - torch.eye(int(n_dim), dtype=torch.float, requires_grad=False)
    optimizer = optim.SGD([n_mean], lr=0.5, momentum=0.9)
    for i in range(max_iters):
        normed_x = n_mean / n_mean.norm(dim=1).unsqueeze(1)
        cov = torch.mm(normed_x.t(), normed_x)**2 / (n_distribution - 1)
        # cov = torch.mm(normed_x.t(), normed_x)**2 / (n_distribution - n_dim)
        loss = torch.mean(cov * temp_matrix)
        loss.backward()
        optimizer.step()
        print(f"Optimizing diverse distribution... n_distribution:{n_distribution}, n_dim:{n_dim}\
                    Iter: {i}, loss: {loss.item()}")
        # logging.debug(f"Optimizing diverse distribution... n_distribution:{n_distribution}, n_dim:{n_dim}\
        #             Iter: {i}, loss: {loss.item()}")

    normed_n_mean = n_mean / n_mean.norm(dim=1).unsqueeze(1)
    return normed_n_mean













