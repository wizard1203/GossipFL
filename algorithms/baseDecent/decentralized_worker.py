import logging
import time
from copy import deepcopy

import torch
from torch import nn
import traceback
from mpi4py import MPI
from utils.data_utils import (
    get_data,
    apply_gradient,
    average_named_params,
    get_local_num_iterations,
    get_avg_num_iterations
)


class BaseDecentralizedWorker(object):
    def __init__(self, client_index, topology_manager, train_data_global, test_data_global, train_data_num,
                train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_number, 
                device, model, args, model_trainer, perf_timer, metrics):
        self.client_index = client_index
        self.in_neighbor_idx_list = topology_manager.get_in_neighbor_idx_list(self.client_index)
        logging.info(self.in_neighbor_idx_list)

        self.worker_result_dict = dict()
        self.flag_neighbor_result_received_dict = dict()
        for neighbor_idx in self.in_neighbor_idx_list:
            self.flag_neighbor_result_received_dict[neighbor_idx] = False

        # Same with fedavg
        # ====================== 
        self.client_index = args.client_index
        self.train_data_global = train_data_global
        self.test_data_global = test_data_global
        self.train_data_num = train_data_num

        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict

        self.test_local = self.test_data_local_dict[client_index]
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]

        self.worker_number = worker_number
        self.device = device
        self.args = args
        self.model_dict = dict()
        self.sample_num_dict = dict()

        self.model_trainer = model_trainer
        self.local_num_iterations, self.global_num_iterations = \
            self.get_num_iterations()


    def get_num_iterations(self):
        local_num_iterations = get_local_num_iterations(self.local_sample_number, self.args.batch_size)
        global_num_iterations = get_avg_num_iterations(self.train_data_local_num_dict, self.args.batch_size)
        return local_num_iterations, global_num_iterations

    def epoch_init(self):
        if self.args.model in ['lstm', 'lstmwt2']:
            self.model_trainer.init_hidden()

    def lr_schedule(self, progress):
        self.model_trainer.lr_schedule(progress)

    def warmup_lr_schedule(self, iterations):
        self.model_trainer.warmup_lr_schedule(iterations)

    # TODO
    def get_batch_len(self):
        return len(self.train_local)

    # TODO
    def get_dataset_len(self):
        return self.local_sample_number

    def set_model_params(self, weights):
        self.model_trainer.set_model_params(weights)

    def get_model_params(self):
        return self.model_trainer.get_model_params()

    def get_train_batch_data(self):
        try:
            train_batch_data = self.train_local_iter.next()
        except:
            self.train_local_iter = iter(self.train_local)
            train_batch_data = self.train_local_iter.next()
        return train_batch_data


    def update_dataset(self):
        # remember to update_dataset every epoch, in order to generate local data iterater.
        # discard this, use try, except to generate iterator.
        # self.train_local_iter = iter(self.train_local)
        pass

    def add_result(self, client_index, updated_information):
        self.worker_result_dict[client_index] = updated_information
        self.flag_neighbor_result_received_dict[client_index] = True

    def check_whether_all_receive(self):
        for neighbor_idx in self.in_neighbor_idx_list:
            if not self.flag_neighbor_result_received_dict[neighbor_idx]:
                return False
        for neighbor_idx in self.in_neighbor_idx_list:
            self.flag_neighbor_result_received_dict[neighbor_idx] = False
        return True

    def check_whether_any_receive(self):
        for neighbor_idx in self.in_neighbor_idx_list:
            if self.flag_neighbor_result_received_dict[neighbor_idx]:
                return True
        return False


    def aggregate(self):
        start_time = time.time()
        model_list = []
        training_num = 0

        #  TODO, There are some bugs
        model_list.append((self.local_sample_number, self.get_model_params()))

        for neighbor_idx in self.in_neighbor_idx_list:
            model_list.append((self.train_data_local_num_dict[neighbor_idx], self.worker_result_dict[neighbor_idx]))
            training_num += self.train_data_local_num_dict[neighbor_idx]

        training_num += self.local_sample_number
        logging.debug("len of self.worker_result_dict[idx] = " + str(len(self.worker_result_dict)))

        # logging.debug("################aggregate: %d" % len(model_list))
        (num0, averaged_params) = model_list[0]
        for k in averaged_params.keys():
            # averaged_params[k] = averaged_params[k] * self.local_sample_number / training_num
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        end_time = time.time()
        logging.debug("aggregate time cost: %d" % (end_time - start_time))
        return averaged_params

    def aggregate_tensor(self, consensus_tensor):
        start_time = time.time()
        model_list = []
        training_num = 0

        logging.debug("begin aggregate_tensor")

        model_list.append((self.local_sample_number, consensus_tensor))

        logging.debug("finish aggregate_tensor")

        for neighbor_idx in self.in_neighbor_idx_list:
            # model_list.append((self.train_data_local_num_dict[neighbor_idx], self.model_dict[neighbor_idx]))
            model_list.append((self.train_data_local_num_dict[neighbor_idx], self.worker_result_dict[neighbor_idx]))
            training_num += self.train_data_local_num_dict[neighbor_idx]

        logging.debug("finish append neighbor tensor")

        training_num += self.local_sample_number
        logging.debug("len of self.worker_result_dict[idx] = " + str(len(self.worker_result_dict)))

        (num0, averaged_params) = model_list[0]
        for i in range(0, len(model_list)):
            local_sample_number, local_model_params = model_list[i]
            w = local_sample_number / training_num
            if i == 0:
                averaged_params = local_model_params * w
            else:
                averaged_params += local_model_params * w

        end_time = time.time()
        logging.debug("aggregate time cost: %d" % (end_time - start_time))
        return averaged_params

    def train_one_step(self, epoch=None, iteration=None, end_of_epoch=False,
            tracker=None, metrics=None
        ):
        train_batch_data = self.get_train_batch_data()
        loss, pred, target = self.model_trainer.train_one_step(
            train_batch_data, device=self.device, args=self.args,
            epoch=epoch, iteration=iteration, end_of_epoch=end_of_epoch,
            tracker=tracker, metrics=metrics)

        return loss, pred, target


    def infer_bw_one_step(self, epoch=None, iteration=None, end_of_epoch=False,
            tracker=None, metrics=None
        ):
        train_batch_data = self.get_train_batch_data()
        loss, pred, target = self.model_trainer.infer_bw_one_step(
            train_batch_data, device=self.device, args=self.args,
            epoch=epoch, iteration=iteration, end_of_epoch=end_of_epoch,
            tracker=tracker, metrics=metrics)

        return loss, pred, target



    def test(self, epoch, tracker, metrics):
        self.model_trainer.test(self.test_local, self.device, self.args, epoch, tracker, metrics)








