import logging

from algorithms.basePS.ps_client_trainer import PSTrainer


from compression.compression import compressors
from utils.perf_timer import Perf_Timer
from utils.tracker import RuntimeTracker
from utils.metrics import Metrics
from utils.data_utils import (
    get_name_params_difference
)

class PSGDTrainer(PSTrainer):

    def __init__(self, client_index, train_data_local_dict, train_data_local_num_dict, test_data_local_dict,
                 train_data_num, device, args, model_trainer, timer, metrics):
        super().__init__(client_index, train_data_local_dict, train_data_local_num_dict, test_data_local_dict,
                 train_data_num, device, args, model_trainer, timer, metrics)
