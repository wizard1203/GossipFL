import logging
import time
from copy import deepcopy

import torch
from torch import nn
import traceback
from mpi4py import MPI

from algorithms.baseDecent.decentralized_worker import BaseDecentralizedWorker

from utils.timer import Timer
# from fedml_api.utils.timer_with_cuda import Timer
from utils.data_utils import (
    get_data,
    apply_gradient
)
from utils.tensor_buffer import (
    TensorBuffer
)


class DecentralizedWorker(BaseDecentralizedWorker):
    def __init__(self, worker_index, topology_manager, train_data_global, test_data_global, train_data_num,
                 train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_number, 
                 device, model, args, model_trainer, timer, metrics):
        super().__init__(worker_index, topology_manager, train_data_global, test_data_global, train_data_num,
                 train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_number, 
                 device, model, args, model_trainer, timer, metrics)
        """
            The `compression` method should be specified in `args`.
        """

        # =================================================
        # Specilaized for DCD
        # ============================================================
        self.param_groups = self.model_trainer.param_groups
        self.param_names = self.model_trainer.param_names
        self.topology_manager = topology_manager
        self.neighbors_info = self.topology_manager.topology

        # will be initialized in init_neighbor_hat_params()
        self.neighbor_hat_params = None
        self.shapes = None
        self.init_neighbor_hat_params()

        # TODO
        # self.compression = args.compression
        # self.compressor = DCDCompressor(self.compression)

    #@override
    def aggregate(self, compressor, selected_shapes):
        start_time = time.time()
        model_list = []
        training_num = 0

        for neighbor_idx in self.in_neighbor_idx_list:
            msg_params = self.worker_result_dict[neighbor_idx]
            compressor.original_shapes = self.shapes
            compressor.uncompress(msg_params,
                self.neighbor_hat_params[neighbor_idx],
                selected_shapes, self.shapes
            )

        logging.debug("len of self.worker_result_dict[idx] = " + str(len(self.worker_result_dict)))

        end_time = time.time()
        logging.debug("aggregate time cost: %d" % (end_time - start_time))
        # return averaged_params


    # ==============================================================================
    # Specilaized for DCD

    def init_neighbor_hat_params(self):
        params, self.shapes = get_data(
            self.param_groups, self.param_names, is_get_grad=False
        )
        flatten_params = TensorBuffer(params)
        flatten_params.buffer = flatten_params.buffer.to(self.device)

        # init the neighbor_params.
        self.neighbor_hat_params = dict()
        # for rank, _ in self.neighbors_info.items():
        for neighbor_idx in self.in_neighbor_idx_list:
            self.neighbor_hat_params[neighbor_idx] = deepcopy(flatten_params)
            self.neighbor_hat_params[neighbor_idx].buffer = \
                self.neighbor_hat_params[neighbor_idx].buffer.to(flatten_params.buffer.device)
            logging.debug("self.neighbor_hat_params[{}].buffer.device : {} \
                flatten_params.buffer.device: {} ".format(
                neighbor_idx,
                self.neighbor_hat_params[neighbor_idx].buffer.device,
                flatten_params.buffer.device
            ))
