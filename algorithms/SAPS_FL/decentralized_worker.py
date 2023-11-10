import logging
import time
from copy import deepcopy

import torch
from torch import nn
import traceback
from mpi4py import MPI

from utils.timer import Timer
# from fedml_api.utils.timer_with_cuda import Timer
from utils.data_utils import (
    get_data,
    apply_gradient
)
from utils.tensor_buffer import (
    TensorBuffer
)

from algorithms.baseDecent.decentralized_worker import BaseDecentralizedWorker

class DecentralizedWorker(BaseDecentralizedWorker):
    def __init__(self, worker_index, topology_manager, train_data_global, test_data_global, train_data_num,
                 train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_number, 
                 device, model, args, model_trainer, timer, metrics):
        """
            The `compression` method should be specified in `args`.
        """
        self.worker_index = worker_index
        self.topology_manager = topology_manager
        self.refresh_gossip_info()
        #===========================================================================
        super().__init__(worker_index, topology_manager, train_data_global, test_data_global, train_data_num,
                 train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_number, 
                 device, model, args, model_trainer, timer, metrics)

        # =================================================
        # Specilaized for SAPS_FL
        # ============================================================
        self.param_groups = self.model_trainer.param_groups
        self.param_names = self.model_trainer.param_names
        # self.neighbors_info = self.topology_manager.topology
        # self.gossip_info = self.topology_manager.topology[self.worker_index]

        # will be initialized in init_neighbor_hat_params()
        self.neighbor_hat_params = None
        self.shapes = None
        # self.init_neighbor_hat_params()


    def aggregate(self, compressor, selected_shapes, gossip_info):
        start_time = time.time()
        model_list = []
        training_num = 0

        for neighbor_idx in self.in_neighbor_idx_list:

            msg_params = self.worker_result_dict[neighbor_idx]
            compressor.uncompress(msg_params, gossip_info[neighbor_idx],
                self.neighbor_hat_params["memory"],
                selected_shapes, self.shapes,
                self.device
            )

        logging.debug("len of self.in_neighbor_idx_list[{}] = {}".format(
            self.worker_index, str(len(self.in_neighbor_idx_list))))
        logging.debug("len of self.worker_result_dict[{}] = {}".format(
            self.worker_index, str(len(self.worker_result_dict))))
        # clear dict for saving memory
        self.worker_result_dict = {}

        end_time = time.time()
        # logging.debug("aggregate time cost: %d" % (end_time - start_time))
        # return averaged_params

    # ==============================================================================
    # Specilaized for SAPS_FL

    def init_neighbor_hat_params(self):
        params, self.shapes = get_data(
            self.param_groups, self.param_names, is_get_grad=False
        )
        flatten_params = TensorBuffer(params)
        flatten_params.buffer = flatten_params.buffer * self.gossip_info[self.worker_index]
        # init the neighbor_params.
        self.neighbor_hat_params = {
            "memory": deepcopy(flatten_params),
            }
        # logging.debug("###################################")
        # logging.debug("self.neighbor_hat_params is on device: {}".format(
        #     self.neighbor_hat_params["memory"].buffer.device
        # ))
        # logging.debug("###################################")

    def refresh_gossip_info(self):
        self.neighbors_info = self.topology_manager.topology
        self.gossip_info = self.topology_manager.topology[self.worker_index]
        self.in_neighbor_idx_list = self.topology_manager.get_in_neighbor_idx_list(self.worker_index)









