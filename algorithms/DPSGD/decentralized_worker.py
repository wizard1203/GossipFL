import logging
import time
from copy import deepcopy

import torch
from torch import nn
import traceback
from mpi4py import MPI


from algorithms.baseDecent.decentralized_worker import BaseDecentralizedWorker



from utils.timer import Timer


class DecentralizedWorker(BaseDecentralizedWorker):
    def __init__(self, worker_index, topology_manager, train_data_global, test_data_global, train_data_num,
                 train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_number, 
                 device, model, args, model_trainer, timer, metrics):
        super().__init__(worker_index, topology_manager, train_data_global, test_data_global, train_data_num,
                 train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_number, 
                 device, model, args, model_trainer, timer, metrics)


