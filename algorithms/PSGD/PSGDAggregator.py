import copy
import logging
import time

import numpy as np
import wandb

from algorithms.basePS.ps_aggregator import PSAggregator

from utils.data_utils import (
    get_data,
    apply_gradient,
    average_named_params,
    idv_average_named_params,
    get_name_params_difference,
    get_local_num_iterations,
    get_avg_num_iterations,
    calc_client_divergence,
    check_device,
    check_type,
)

from utils.tensor_buffer import (
    TensorBuffer
)

from compression.compression import compressors


class PSGDAggregator(PSAggregator):
    def __init__(self, train_global, test_global, all_train_data_num,
                train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_num, device,
                args, model_trainer, perf_timer, metrics):
        super().__init__(train_global, test_global, all_train_data_num,
                train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_num, device,
                args, model_trainer, perf_timer, metrics)
        if self.args.psgd_exchange == 'grad':
            self.train_global_iter = iter(self.train_global)
            self.init_for_generate_fake_grad()

    def aggregate_grads(self, grad_list, client_other_params_list, sample_num_list, training_num,
                        global_comm_round=0, global_outer_epoch_idx=0):
        start_time = time.time()

        average_weights_dict_list, homo_weights_list = self.get_average_weight_dict(
            sample_num_list=sample_num_list,
            client_other_params_list=client_other_params_list,
            global_comm_round=global_comm_round,
            global_outer_epoch_idx=global_outer_epoch_idx)

        averaged_params = average_named_params(
            grad_list,
            average_weights_dict_list
            )

        # update the global model which is cached at the server side
        # In grad averaging way, do not update models here.

        end_time = time.time()
        logging.debug("aggregate time cost: %d" % (end_time - start_time))
        return averaged_params





    def aggregate_bn_params(self, client_other_params_list, sample_num_list, training_num,
                        global_comm_round=0, global_outer_epoch_idx=0):
        start_time = time.time()

        bn_params_list = [client_other_params["all_bn_params"] for client_other_params in client_other_params_list]

        average_weights_dict_list, homo_weights_list = self.get_average_weight_dict(
            sample_num_list=sample_num_list,
            client_other_params_list=client_other_params_list,
            global_comm_round=global_comm_round,
            global_outer_epoch_idx=global_outer_epoch_idx)

        averaged_params = average_named_params(
            bn_params_list,
            average_weights_dict_list
            )

        # update the global model which is cached at the server side
        # In grad averaging way, do not update models here.

        end_time = time.time()
        logging.debug("aggregate time cost: %d" % (end_time - start_time))
        return averaged_params



