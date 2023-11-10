import logging
import os
import sys
from abc import ABC, abstractmethod

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))


from fedml_core.distributed.communication.message import Message
from fedml_core.distributed.server.server_manager import ServerManager

from utils.timer import Timer
from utils.tracker import RuntimeTracker
from utils.metrics import Metrics
from utils.wandb_util import wandb_log, upload_metric_info

from .message_define import MyMessage


class PSServerManager(ServerManager):
    def __init__(self, args, aggregator, comm=None, rank=0, size=0, backend="MPI", timer=None, metrics=None):
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.aggregator = aggregator
        self.global_num_iterations = self.aggregator.global_num_iterations
        self.comm_round = self.get_comm_round()
        self.global_round_idx = 0
        self.iteration = 0
        self.epoch = 0
        # assert args.client_num_in_total == self.size - 1
        # assert args.client_num_per_round == self.size - 1
        self.selected_clients = None
        # ================================================
        self.metrics = metrics
        self.total_train_tracker = RuntimeTracker(things_to_track=self.metrics.metric_names)
        self.total_test_tracker = RuntimeTracker(things_to_track=self.metrics.metric_names)


    def run(self):
        super().run()

    @abstractmethod
    def get_comm_round(self):
        """
            Maybe this is only useful in FedAvg.
        """
        pass

    def epoch_init(self):
        self.aggregator.epoch_init()

    def update_time(self):
        """
            Remember to revise them if needed.
        """
        self.global_round_idx += 1
        self.epoch = self.global_round_idx // self.global_num_iterations
        self.iteration = self.global_round_idx % self.global_num_iterations

    def send_init_msg(self):
        # sampling clients
        logging.debug("send_init_msg")

        client_indexes = self.aggregator.client_sampling(self.global_round_idx, self.args.client_num_in_total,
                                                         self.args.client_num_per_round)
        self.selected_clients = client_indexes
        global_model_params = self.aggregator.get_global_model_params()
        for process_id in client_indexes:
            self.send_message_init_config(process_id+1, global_model_params, process_id,
                                          global_round_idx=self.global_round_idx)

    def get_timestamp_from_msg(self, msg_params):
        global_round_idx = msg_params.get(MyMessage.MSG_ARG_KEY_GLOBAL_ROUND_INDEX)
        local_round_idx = msg_params.get(MyMessage.MSG_ARG_KEY_LOCAL_ROUND_INDEX)
        local_epoch_idx = msg_params.get(MyMessage.MSG_ARG_KEY_LOCAL_EPOCH_INDEX)
        local_iter_idx = msg_params.get(MyMessage.MSG_ARG_KEY_LOCAL_ITER_INDEX)
        local_total_iter_idx = msg_params.get(MyMessage.MSG_ARG_KEY_LOCAL_TOTAL_ITER_INDEX)
        return global_round_idx, local_round_idx, local_epoch_idx, local_iter_idx, local_total_iter_idx


    def get_metric_info_from_message(self, msg_params):
        local_train_metric_info = msg_params.get(MyMessage.MSG_ARG_KEY_TRAIN_METRICS)
        local_test_metric_info = msg_params.get(MyMessage.MSG_ARG_KEY_TEST_METRICS)

        if local_train_metric_info is not None:
            logging.debug('Server: receive train_metric_info')
            assert local_train_metric_info['n_samples'] > 0
            self.total_train_tracker.update_metrics(local_train_metric_info, local_train_metric_info['n_samples'])

        if local_test_metric_info is not None:
            logging.debug('Server: receive test_metric_info')
            assert local_test_metric_info['n_samples'] > 0
            self.total_test_tracker.update_metrics(local_test_metric_info, local_test_metric_info['n_samples'])
        return local_train_metric_info, local_test_metric_info


    def reset_train_test_tracker(self, train_trakcer, test_tracker):
        self.total_train_tracker.reset()
        self.total_test_tracker.reset()


    def choose_clients_and_send(self, global_params, params_type='grad', global_round_idx=None):
        # sampling clients
        client_indexes = self.aggregator.client_sampling(self.global_round_idx, self.args.client_num_in_total,
                                                            self.args.client_num_per_round)
        logging.debug("size = %d" % self.size)
        self.selected_clients = client_indexes
        for receiver_id in client_indexes:
            if params_type == 'grad':
                self.send_message_sync_grad_to_client(
                    receiver_id+1, global_params, receiver_id, global_round_idx=global_round_idx)
            elif params_type == 'model':
                self.send_message_sync_model_to_client(
                    receiver_id+1, global_params, receiver_id, global_round_idx=global_round_idx)


    def send_message_init_config(self, receive_id, global_model_params, client_index, global_round_idx=None):
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_GLOBAL_ROUND_INDEX, global_round_idx)
        self.send_message(message)

    def send_message_sync_model_to_client(self, receive_id, global_model_params, client_index, global_round_idx=None):
        logging.debug("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_GLOBAL_ROUND_INDEX, global_round_idx)
        self.send_message(message)

    def send_message_sync_grad_to_client(self, receive_id, global_grad_params, client_index, global_round_idx=None):
        logging.debug("send_message_sync_grad_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_GRAD_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_GRAD_PARAMS, global_grad_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_GLOBAL_ROUND_INDEX, global_round_idx)
        self.send_message(message)









