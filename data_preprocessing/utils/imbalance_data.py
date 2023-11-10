import logging

import torch
import shutil
import os
import numpy as np

from utils.data_utils import get_per_cls_weights

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, args, dataset, indices=None, num_samples=None, class_num=10, **kwargs):
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
        logging.info("self.indices: {}".format(self.indices))
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        self.args = args
        self.dataset = dataset
        # distribution of classes in the dataset 
        # label_to_count = [0] * len(np.unique(dataset.target))
        label_to_count = [0] * class_num
        logging.info("label_to_count: {}".format(label_to_count))
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[label] += 1
        for i in range(len(label_to_count)):
            if label_to_count[i] == 0:
                label_to_count[i] = 1

        self.label_to_count = label_to_count
        self.base_beta = args.imbalance_beta
        self.beta = self.base_beta
        self.imbalance_beta_min = args.imbalance_beta_min
        self.imbalance_beta_decay_rate = args.imbalance_beta_decay_rate
        self.imbalance_beta_decay_type = args.imbalance_beta_decay_type

        effective_num = 1.0 - np.power(self.beta, label_to_count)
        per_cls_weights = (1.0 - self.beta) / np.array(effective_num)

        # weight for each sample
        weights = [per_cls_weights[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.target[idx]

    def update(self, **kwargs):
        exp_num = kwargs[self.imbalance_beta_decay_type]

        if self.args.data_sampler == 'imbalance':
            pass
        elif self.args.data_sampler == "decay_imb":
            self.beta = self.imbalance_beta_min + \
                (self.base_beta - self.imbalance_beta_min) * (self.imbalance_beta_decay_rate**exp_num)
            effective_num = 1.0 - np.power(self.beta, self.label_to_count)
            per_cls_weights = (1.0 - self.beta) / np.array(effective_num)

            # weight for each sample
            weights = [per_cls_weights[self._get_label(self.dataset, idx)]
                    for idx in self.indices]
            self.weights = torch.DoubleTensor(weights)
        else:
            raise NotImplementedError

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples