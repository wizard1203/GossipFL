import logging
import threading
import time
from copy import deepcopy

import torch
import traceback
from mpi4py import MPI

from fedml_core.distributed.client.client_manager import ClientManager
from fedml_core.distributed.communication.message import Message

from .message_define import MyMessage

from utils.context import (
    raise_MPI_error,
    raise_error_without_process,
    get_lock,
)
from utils.timer import Timer
from utils.tracker import RuntimeTracker, get_metric_info
from utils.metrics import Metrics
from utils.wandb_util import wandb_log, upload_metric_info

comm = MPI.COMM_WORLD

class BaseDecentralizedWorkerManager(ClientManager):
    def __init__(self, args, comm, rank, size, worker, topology_manager, model_trainer, timer, metrics):
        super().__init__(args, comm, rank, size)
        self.worker_index = rank
        self.worker = worker
        self.size = size

        self.model_trainer = model_trainer
        self.topology_manager = topology_manager
        self.gossip_info = self.topology_manager.topology[self.worker_index]
        self.comm_round = args.comm_round
        self.global_round_idx = 0

        self.epochs = args.epochs
        self.comm_round = self.get_comm_round()
        self.epoch = 0
        self.iteration = 0

        self.flag_client_finish_dict = dict()
        for client_index in range(self.size):
            self.flag_client_finish_dict[client_index] = False

        self.timer = timer
        self.metrics = metrics
        self.train_tracker = RuntimeTracker(things_to_track=self.metrics.metric_names)
        self.test_tracker = RuntimeTracker(things_to_track=self.metrics.metric_names)

        self.coodinator_upload_results_flag = False
        self.total_test_tracker = None
        self.total_train_tracker = None
        if self.worker_index == 0:
            self.total_test_tracker = RuntimeTracker(things_to_track=self.metrics.metric_names)
            self.total_train_tracker = RuntimeTracker(things_to_track=self.metrics.metric_names)
            self.coodinator_thread = threading.Thread(name="coordinator", target=self.run_coordinator)

        self.training_thread = threading.Thread(name='training', target=self.run_sync)

        # For coordinator notify event
        self.all_clients_finish_event = threading.Event()
        self.total_metric_lock = threading.Lock()
        self.start_epoch_event = threading.Event()


    def get_comm_round(self):
        return self.args.epochs * self.worker.num_iterations

    def epoch_init(self):
        self.worker.epoch_init()


    def update_time(self):
        self.global_round_idx += 1
        self.epoch = self.global_round_idx // self.num_iterations
        self.iteration = self.global_round_idx % self.num_iterations

    def lr_schedule(self, epoch, iteration, round_idx, num_iterations, warmup_epochs):
        if self.args.sched == "no":
            pass
        else:
            if epoch < warmup_epochs:
                self.model_trainer.warmup_lr_schedule(epoch * num_iterations + iteration)
            else:
                # When epoch begins, do lr_schedule.
                if iteration == 0:
                    self.model_trainer.lr_schedule(epoch)

    def handle_msg_client_to_coordinator(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        logging.debug("handle_msg_client_to_coordinator. sender_id = " + str(sender_id))

        with get_lock(self.total_metric_lock):
            logging.debug("get metric lock, handle_msg_client_to_coordinator. sender_id = " + str(sender_id))
            self.flag_client_finish_dict[sender_id] = True

            train_metric_info = msg_params.get(MyMessage.MSG_ARG_KEY_TRAIN_METRICS)
            if train_metric_info is not None and train_metric_info['n_samples'] > 0:
                self.total_train_tracker.update_metrics(train_metric_info, train_metric_info['n_samples'])

            test_metric_info = msg_params.get(MyMessage.MSG_ARG_KEY_TEST_METRICS)
            if test_metric_info is not None and test_metric_info['n_samples'] > 0:
                self.coodinator_upload_results_flag = True
                self.total_test_tracker.update_metrics(test_metric_info, test_metric_info['n_samples'])
            else:
                self.coodinator_upload_results_flag = False

            self.check_worker_finish_and_notify()


    def handle_msg_coordinator_to_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        logging.debug("handle_msg_coordinator_to_client. sender_id = " + str(sender_id))
        self.start_epoch_event.set()


    def test_and_send_to_coordinator(self, iteration, epoch):
        if (iteration == self.worker.num_iterations - 1) and \
        (self.epoch % self.args.frequency_of_the_test == 0 or self.epoch == self.args.epochs - 1):
            # if iteration > 0:
            """ one epoch ends here"""
            # test
            self.worker.test(self.epoch, self.test_tracker, self.metrics)
            # logging.debug('Local Test Epoch: {} \t Loss: {:.6f}, Acc1: {:.6f}'.format(
            #             epoch, self.test_tracker()['Loss'], self.test_tracker()['Acc1']))

            train_metric_info, test_metric_info = get_metric_info(
                    self.train_tracker, self.test_tracker, time_stamp=self.epoch, if_reset=True,
                    metrics=self.metrics)

            if self.rank == 0:
                with get_lock(self.total_metric_lock):
                    self.flag_client_finish_dict[self.rank] = True
                    self.total_test_tracker.update_metrics(test_metric_info, test_metric_info['n_samples'])
                    self.total_train_tracker.update_metrics(train_metric_info, train_metric_info['n_samples'])
                    self.check_worker_finish_and_notify()
            else:
                self.send_notify_to_coordinator(0, train_metric_info, test_metric_info)

        else:
            self.reset_train_test_tracker(self.train_tracker, self.test_tracker)
            if self.rank == 0:
                with get_lock(self.total_metric_lock):
                    self.flag_client_finish_dict[self.rank] = True
                    self.check_worker_finish_and_notify()
            else:
                self.send_notify_to_coordinator(0)


    def reset_train_test_tracker(self, train_tracker, test_tracker):
        train_tracker.reset()
        test_tracker.reset()



    def run_coordinator(self):
        with raise_MPI_error():
            for epoch in range(self.args.epochs):
                for iteration in range(self.worker.num_iterations):
                    self.all_clients_finish_event.wait()
                    if self.coodinator_upload_results_flag:
                        train_metric_info, test_metric_info = get_metric_info(
                            self.total_train_tracker, self.total_test_tracker, time_stamp=self.epoch, if_reset=True,
                            metrics=self.metrics)
                        com_values = {"epoch": train_metric_info['time_stamp'], "round": self.global_round_idx}
                        upload_metric_info(str_pre="Epoch: {}, Server Total: ".format(train_metric_info['time_stamp']),
                                           train_metric_info=train_metric_info,
                                        test_metric_info=test_metric_info, metrics=self.metrics,
                                        comm_values=com_values)
                        # logging.debug('(Global Training Epoch: {}, Iter: {} '.format(
                        #     epoch, iteration) + self.metrics.str_fn(train_metric_info))
                        # logging.debug('(Global Testing Epoch: {}, Iter: {} '.format(
                        #     epoch, iteration) + self.metrics.str_fn(test_metric_info))
                    else:
                        pass
                    self.all_clients_finish_event.clear()


    def check_whether_all_clients_finish_receive(self):
        for rank, flag in self.flag_client_finish_dict.items():
            if not flag:
                return False
        for rank, _ in self.flag_client_finish_dict.items():
            self.flag_client_finish_dict[rank] = False
        return True


    def check_worker_finish_and_notify(self):
        if self.check_whether_all_clients_finish_receive():
            logging.debug(">>>>>>>>>>>>>>>COORDINATOR Receive all, ROUND %d finished!<<<<<<<<" %
                 (self.global_round_idx))
            self.all_clients_finish_event.set()
            self.notify_clients()
            if self.global_round_idx == self.comm_round:
                self.finish()


    def notify_clients(self):
        logging.debug("COORDINATOR notify clients to start!")
        for client_index in range(self.size):
            if client_index == 0:
                self.start_epoch_event.set()
            else:
                self.send_notify_to_clients(client_index)


    def send_message_init_config(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_INIT, self.get_sender_id(), receive_id)
        self.send_message(message)

    def send_result_to_neighbors(self, receive_id, client_params1, local_sample_number):
        logging.debug("send_result_to_neighbors. receive_id = %s, round: %s" % (str(receive_id), str(self.global_round_idx)))
        message = Message(MyMessage.MSG_TYPE_SEND_MSG_TO_NEIGHBOR, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_PARAMS_1, client_params1)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_number)
        self.send_message(message)

    def send_notify_to_coordinator(self, receive_id=0, train_metric_info=None, test_metric_info=None):
        logging.debug("send_notify_to_coordinator. receive_id = %s, round: %s" % (str(receive_id), str(self.global_round_idx)))
        message = Message(MyMessage.MSG_TYPE_CLIENT_TO_COORDINATOR, self.get_sender_id(), receive_id)

        # train
        message.add_params(MyMessage.MSG_ARG_KEY_TRAIN_METRICS, train_metric_info)

        # test
        message.add_params(MyMessage.MSG_ARG_KEY_TEST_METRICS, test_metric_info)
        self.send_message(message)

    def send_notify_to_clients(self, receive_id):
        logging.debug("send_notify_to_clients. receive_id = %s, round: %s" % (str(receive_id), str(self.global_round_idx)))
        message = Message(MyMessage.MSG_TYPE_COORDINATOR_TO_CLIENT, self.get_sender_id(), receive_id)
        self.send_message(message)
