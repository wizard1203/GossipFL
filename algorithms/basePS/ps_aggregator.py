import copy
import logging
import time

import numpy as np
import wandb


from utils.timer import Timer
from utils.tracker import RuntimeTracker
from utils.metrics import Metrics
from utils.wandb_util import wandb_log
from utils.data_utils import (
    get_data,
    apply_gradient,
    average_named_params,
    get_name_params_difference,
    get_local_num_iterations,
    get_avg_num_iterations
)
from utils.tensor_buffer import (
    TensorBuffer
)

from compression.compression import compressors


class PSAggregator(object):
    def __init__(self, train_global, test_global, all_train_data_num,
                 train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_num, device,
                 args, model_trainer, timer, metrics):
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

        # this flag_client_model_uploaded_dict flag dict is commonly used by gradient and model params
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False

        self.selected_clients = None
        # ====================================
        self.timer = timer
        # ====================================
        self.global_num_iterations = self.get_num_iterations()
        if args.compression is None or self.args.compression == 'no':
            pass
        else:
            self.compressor = compressors[args.compression]()
            model_params = self.get_global_model_params()
            for k in model_params.keys():
                self.compressor.update_shapes_dict(model_params[k], k)

    def get_num_iterations(self):
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


    def add_local_trained_result(self, index, model_params, model_indexes, sample_num):
        """
            Note: For APSGD and SSPSGD, this function is overrided, due to asynchronous updates.
        """
        logging.debug("add_model. index = %d" % index)
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

        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def add_local_trained_grad(self, index, grad_params, grad_indexes, sample_num):
        """
            Note: For APSGD and SSPSGD, this function is overrided, due to asynchronous updates.
        """
        logging.debug("add_grad. index = %d" % index)
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

        self.grad_dict[index] = grad_params
        self.sample_num_dict[index] = sample_num
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


    def aggregate(self):
        start_time = time.time()
        model_list = []
        training_num = 0
        if self.args.if_get_diff is True and self.args.psgd_exchange == "model":
            logging.debug("Server is averaging model diff!!")
            averaged_params = self.get_global_model_params()
            for idx in range(self.worker_num):
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
            for idx in range(self.worker_num):
                model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
                training_num += self.sample_num_dict[idx]

            logging.debug("len of self.model_dict[idx] = " + str(len(self.model_dict)))

            averaged_params = average_named_params(model_list, training_num)
        else:
            raise NotImplementedError

        # update the global model which is cached at the server side
        self.set_global_model_params(averaged_params)

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return averaged_params

