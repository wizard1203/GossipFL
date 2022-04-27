import copy
import logging
import time

from numpy.core.getlimits import _register_type

import torch
import torch.optim as optim
import torch.nn.functional as F


import numpy as np
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import platform
sysstr = platform.system()
if ("Windows" in sysstr):
    matplotlib.use("TkAgg")
    logging.info ("On Windows, matplotlib use TkAgg")
else:
    matplotlib.use("Agg")
    logging.info ("On Linux, matplotlib use Agg")

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

from utils.checkpoint import (
    setup_checkpoint_file_name_prefix, save_images
)
from compression.compression import compressors

from timers.server_timer import ServerTimer



class PSAggregator(object):
    def __init__(self, train_global, test_global, all_train_data_num,
                train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_num, device,
                args, model_trainer, perf_timer, metrics):
        self.trainer = model_trainer

        self.train_global = train_global

        self.test_global = test_global
        self.all_train_data_num = all_train_data_num

        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict

        self.worker_num = worker_num
        self.device = device
        self.args = args
        self.model_dict = dict()
        self.grad_dict = dict()
        self.sample_num_dict = dict()

        # Saving the client_other_params of clients
        self.client_other_params_dict = dict()

        # this flag_client_model_uploaded_dict flag dict is commonly used by gradient and model params
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False

        self.selected_clients = None
        # ====================================
        self.perf_timer = perf_timer
        # ====================================
        self.global_num_iterations = self.get_num_iterations()
        if args.compression is None or self.args.compression == 'no':
            pass
        else:
            self.compressor = compressors[args.compression]()
            model_params = self.get_global_model_params()
            for k in model_params.keys():
                self.compressor.update_shapes_dict(model_params[k], k)

        if self.args.pretrained:
            ckt = torch.load(args.pretrained_dir)
            self.trainer.model.load_state_dict(ckt["model_state_dict"])


    def get_num_iterations(self):
        # return 20
        return get_avg_num_iterations(self.train_data_local_num_dict, self.args.batch_size)

    def epoch_init(self):
        if self.args.model in ['lstm', 'lstmwt2']:
            self.trainer.init_hidden()


    def get_global_model_params(self):
        return self.trainer.get_model_params()

    def set_global_model_params(self, model_parameters):
        self.trainer.set_model_params(model_parameters)

    def set_grad_params(self, named_grads):
        self.trainer.set_grad_params(named_grads)

    def clear_grad_params(self):
        self.trainer.clear_grad_params()

    def update_model_with_grad(self):
        self.trainer.update_model_with_grad()


    def uncompress_model_params(self, model_params, model_indexes):
        if model_indexes is not None:
            for k in model_indexes.keys():
                # logging.debug("model_params[k]:{}, model_indexes[k]:{}, k:{}".format(
                #     model_params[k], model_indexes[k], k
                # ))
                model_params[k] = self.compressor.unflatten(
                    self.compressor.decompress_new(model_params[k], model_indexes[k], k), k)
        elif self.args.compression is not None and self.args.compression != 'no':
            # TODO, add quantize here
            for k in model_params.keys():
                # logging.debug("model_params[k]:{}, model_indexes[k]:{}, k:{}".format(
                #     model_params[k], model_indexes[k], k
                # ))
                model_params[k] = self.compressor.decompress_new(model_params[k])
        else:
            pass
        return model_params

    def uncompress_grad_params(self, grad_params, grad_indexes):
        if grad_indexes is not None:
            for k in grad_indexes.keys():
                # logging.debug("model_params[k]:{}, model_indexes[k]:{}, k:{}".format(
                #     model_params[k], model_indexes[k], k
                # ))
                grad_params[k] = self.compressor.unflatten(
                    self.compressor.decompress_new(grad_params[k], grad_indexes[k], k), k)
        elif self.args.compression is not None and self.args.compression != 'no':
            # TODO, add quantize here
            for k in grad_params.keys():
                # logging.debug("model_params[k]:{}, model_indexes[k]:{}, k:{}".format(
                #     model_params[k], model_indexes[k], k
                # ))
                grad_params[k] = self.compressor.decompress_new(grad_params[k])
        else:
            pass
        return grad_params


    def add_local_trained_result(self, index, model_params, model_indexes, sample_num, 
            client_other_params=None):
        """
            Note: For APSGD and SSPSGD, this function is overrided, due to asynchronous updates.
        """
        logging.debug("add_model. index = %d" % index)
        model_params = self.uncompress_model_params(model_params, model_indexes)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.client_other_params_dict[index] = client_other_params
        self.flag_client_model_uploaded_dict[index] = True

    def add_local_trained_grad(self, index, grad_params, grad_indexes, sample_num,
            client_other_params=None):
        """
            Note: For APSGD and SSPSGD, this function is overrided, due to asynchronous updates.
        """
        logging.debug("add_grad. index = %d" % index)
        grad_params = self.uncompress_grad_params(grad_params, grad_indexes)
        self.grad_dict[index] = grad_params
        self.sample_num_dict[index] = sample_num
        self.client_other_params_dict[index] = client_other_params
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("sampling client_indexes = %s" % str(client_indexes))
        self.selected_clients = client_indexes
        return client_indexes


    def test_on_server_for_all_clients(self, epoch, tracker=None, metrics=None):
        logging.info("################test_on_server_for_all_clients : {}".format(epoch))
        self.trainer.test(self.test_global, self.device, self.args, epoch, tracker, metrics)
        # self.trainer.test(self.train_global, self.device, self.args, epoch, tracker, metrics)


    def get_average_weight_dict(self, sample_num_list, client_other_params_list,
            global_comm_round=0, global_outer_epoch_idx=0):

        # if self.args.algorithm == "FedAvg":
        #     avg_weight_type = self.args.fedavg_avg_weight_type
        # else:
        #     raise NotImplementedError
        avg_weight_type = self.args.fedavg_avg_weight_type
        average_weights_dict_list, homo_weights_list = \
            self.trainer.averager.get_average_weight(
                sample_num_list, avg_weight_type, global_outer_epoch_idx, inplace=True)
            # raise NotImplementedError
        return average_weights_dict_list, homo_weights_list


    def get_received_params_list(self, params_type='model'):
        params_list = []
        sample_num_list = []
        training_num = 0
        client_other_params_list = []
        
        if params_type == 'model':
            params_dict = self.model_dict
        elif params_type == 'grad':
            params_dict = self.grad_dict
        else:
            raise NotImplementedError

        for idx in self.selected_clients:
            params_list.append((self.sample_num_dict[idx], params_dict[idx]))
            sample_num_list.append(self.sample_num_dict[idx])
            if idx in self.client_other_params_dict:
                client_other_params = self.client_other_params_dict[idx]
            else:
                client_other_params = {}
            client_other_params_list.append(client_other_params)
            training_num += self.sample_num_dict[idx]
        return params_list, client_other_params_list, sample_num_list, training_num


    def aggregate(self, global_comm_round=0, global_outer_epoch_idx=0, tracker=None, metrics=None,
                ):
        start_time = time.time()
        model_list = []
        training_num = 0

        global_other_params = {}
        shared_params_for_simulation = {}

        if self.args.if_get_diff is True and self.args.psgd_exchange == "model":
            logging.debug("Server is averaging model diff!!")
            averaged_params = self.get_global_model_params()
            # for idx in range(self.worker_num):
            for idx in self.selected_clients:
                model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
                training_num += self.sample_num_dict[idx]

            logging.debug("len of self.model_dict[idx] = " + str(len(self.model_dict)))

            for k in averaged_params.keys():
                for i in range(0, len(model_list)):
                    local_sample_number, local_model_params = model_list[i]
                    w = local_sample_number / training_num
                    # logging.info("averaged_params[k].dtype: {}, local_model_params[k].dtype: {}".format(
                    #     averaged_params[k].dtype, local_model_params[k].dtype
                    # ))
                    averaged_params[k] += (local_model_params[k] * w).type(averaged_params[k].dtype)
        elif self.args.if_get_diff is False:
            logging.debug("Server is averaging model or adding grads!!")
            # for idx in range(self.worker_num):
            sample_num_list = []
            client_other_params_list = []
            for idx in self.selected_clients:
                model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
                sample_num_list.append(self.sample_num_dict[idx])
                if idx in self.client_other_params_dict:
                    client_other_params = self.client_other_params_dict[idx]
                else:
                    client_other_params = {}
                client_other_params_list.append(client_other_params)
                training_num += self.sample_num_dict[idx]

            logging.debug("len of self.model_dict[idx] = " + str(len(self.model_dict)))
            logging.info("Aggregator: using average type: {} ".format(
                self.args.fedavg_avg_weight_type
            ))

            average_weights_dict_list, homo_weights_list = self.get_average_weight_dict(
                sample_num_list=sample_num_list,
                client_other_params_list=client_other_params_list,
                global_comm_round=global_comm_round,
                global_outer_epoch_idx=global_outer_epoch_idx)

            averaged_params = average_named_params(
                model_list,
                average_weights_dict_list
                )

        else:
            raise NotImplementedError

        # update the global model which is cached at the server side
        self.set_global_model_params(averaged_params)
        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return averaged_params, global_other_params, shared_params_for_simulation



    def init_for_generate_fake_grad(self, tracker=None, metrics=None):

        self.epoch_init()
        train_batch_data = next(self.train_global_iter)
        loss, pred, target = self.trainer.infer_bw_one_step(
            train_batch_data, device=self.device, args=self.args, 
            tracker=tracker, metrics=metrics)
        logging.info("init_for_generate_fake_grad")






