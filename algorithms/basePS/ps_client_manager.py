import logging
import os
import sys
from abc import ABC, abstractmethod

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))


from fedml_core.distributed.client.client_manager import ClientManager
from fedml_core.distributed.communication.message import Message

from utils.timer import Timer
from utils.tracker import RuntimeTracker
from utils.metrics import Metrics
from utils.wandb_util import wandb_log

from utils.context import (
    raise_MPI_error,
    raise_error_without_process,
    get_lock,
)

from .message_define import MyMessage


class PSClientManager(ClientManager):
    def __init__(self, args, trainer, comm=None, rank=0, size=0, backend="MPI", timer=None, metrics=None):
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.trainer = trainer
        # self.num_iterations = self.trainer.num_iterations
        self.local_num_iterations = self.trainer.local_num_iterations
        self.global_num_iterations = self.trainer.global_num_iterations
        self.comm_round = self.get_comm_round()
        self.global_round_idx = 0
        self.local_round_idx = -1
        self.epoch = 0
        self.iteration = 0
        self.total_iteration = 0
        # ================================================
        self.metrics = metrics
        self.train_tracker = RuntimeTracker(things_to_track=self.metrics.metric_names)
        self.test_tracker = RuntimeTracker(things_to_track=self.metrics.metric_names)

    def run(self):
        super().run()

    @abstractmethod
    def get_comm_round(self):
        """
            Implement in specific algorithms.
            Maybe it is only needed in FedAvg.
        """
        pass

    def epoch_init(self):
        self.trainer.epoch_init()

    def get_metric_info(self, time_stamp, if_reset):
        self.test_tracker.update_time_stamp(time_stamp=time_stamp)
        test_metric_info = self.test_tracker()
        self.train_tracker.update_time_stamp(time_stamp=time_stamp)
        train_metric_info = self.train_tracker()

        if if_reset:
            self.reset_train_test_tracker()
        else:
            logging.info("WARNING: train_tracker and test_tracker are not reset!!!")
        return test_metric_info, train_metric_info

    def reset_train_test_tracker(self):
        self.train_tracker.reset()
        self.test_tracker.reset()

    def update_time(self, global_round_idx):
        """
            Remember to revise them if needed.
        """
        self.global_round_idx = global_round_idx
        self.local_round_idx = self.global_round_idx
        self.epoch = self.local_round_idx // self.global_num_iterations
        self.iteration = self.local_round_idx % self.global_num_iterations
        self.total_iteration = self.local_round_idx


    def lr_schedule(self, epoch, iteration, round_idx, num_iterations, warmup_epochs):
        if self.args.sched == "no":
            pass
        else:
            if epoch < warmup_epochs:
                self.trainer.warmup_lr_schedule(epoch * num_iterations + iteration)
            else:
                # When epoch begins, do lr_schedule.
                if (round_idx > 0 and round_idx % num_iterations == 0):
                    self.trainer.lr_schedule(epoch)


    def send_model_to_server(self, receive_id, weights, local_sample_num, model_indexes=None,
                             global_round_idx=None, local_round_idx=None,
                             local_epoch_idx=None, local_iter_idx=None, local_total_iter_idx=None,
                             train_metric_info=None, test_metric_info=None):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_INDEXES, model_indexes)

        self.add_timestamp_to_msg(message, global_round_idx, local_round_idx,
                        local_epoch_idx, local_iter_idx, local_total_iter_idx)

        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        # train
        message.add_params(MyMessage.MSG_ARG_KEY_TRAIN_METRICS, train_metric_info)
        # test
        message.add_params(MyMessage.MSG_ARG_KEY_TEST_METRICS, test_metric_info)
        self.send_message(message)

    def send_grad_to_server(self, receive_id, grads, local_sample_num, grad_indexes=None,
                            global_round_idx=None, local_round_idx=None,
                            local_epoch_idx=None, local_iter_idx=None, local_total_iter_idx=None,
                            train_metric_info=None, test_metric_info=None):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_GRAD_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_GRAD_PARAMS, grads)
        message.add_params(MyMessage.MSG_ARG_KEY_GRAD_INDEXES, grad_indexes)

        self.add_timestamp_to_msg(message, global_round_idx, local_round_idx,
                        local_epoch_idx, local_iter_idx, local_total_iter_idx)

        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        # train
        message.add_params(MyMessage.MSG_ARG_KEY_TRAIN_METRICS, train_metric_info)
        # test
        message.add_params(MyMessage.MSG_ARG_KEY_TEST_METRICS, test_metric_info)
        self.send_message(message)


    def add_timestamp_to_msg(self, message, global_round_idx=None, local_round_idx=None,
                        local_epoch_idx=None, local_iter_idx=None, local_total_iter_idx=None):
        message.add_params(MyMessage.MSG_ARG_KEY_GLOBAL_ROUND_INDEX, global_round_idx)
        message.add_params(MyMessage.MSG_ARG_KEY_LOCAL_ROUND_INDEX, local_round_idx)
        message.add_params(MyMessage.MSG_ARG_KEY_LOCAL_EPOCH_INDEX, local_epoch_idx)
        message.add_params(MyMessage.MSG_ARG_KEY_LOCAL_ITER_INDEX, local_iter_idx)
        message.add_params(MyMessage.MSG_ARG_KEY_LOCAL_TOTAL_ITER_INDEX, local_total_iter_idx)







