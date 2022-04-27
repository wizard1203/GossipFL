import logging
import os
import sys
import copy
from abc import ABC, abstractmethod

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))


from fedml_core.distributed.client.client_manager import ClientManager
from fedml_core.distributed.communication.message import Message

from utils.tracker import RuntimeTracker
from utils.context import (
    raise_error_without_process,
    get_lock,
)

from .message_define import MyMessage

from timers.client_timer import ClientTimer


class PSClientManager(ClientManager):
    def __init__(self, args, trainer, comm=None, rank=0, size=0, backend="MPI", perf_timer=None, metrics=None):
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.client_index = args.client_index
        self.trainer = trainer
        # self.num_iterations = self.trainer.num_iterations
        self.local_num_iterations = self.trainer.local_num_iterations
        self.global_num_iterations = self.trainer.global_num_iterations
        self.max_comm_round = self.get_max_comm_round()
        self.local_comm_round_idx_dict = {}

        # ================================================
        local_num_iterations_dict = {}
        local_num_iterations_dict[self.client_index] = self.local_num_iterations
        self.client_timer = ClientTimer(
            self.args,
            self.global_num_iterations,
            local_num_iterations_dict
        )
        # ================================================
        self.metrics = metrics
        self.train_tracker = RuntimeTracker(
            mode='Train',
            things_to_metric=self.metrics.metric_names,
            timer=self.client_timer,
            args=args
        )
        self.test_tracker = RuntimeTracker(
            mode='Test',
            things_to_metric=self.metrics.metric_names,
            timer=self.client_timer,
            args=args
        )

    def run(self):
        super().run()

    @abstractmethod
    def get_max_comm_round(self):
        """
            Implement in specific algorithms.
            Maybe it is only needed in FedAvg.
        """
        pass

    def epoch_init(self):
        self.trainer.epoch_init()

    def check_end_epoch(self):
        return (self.client_timer.local_outer_iter_idx > 0 and self.client_timer.local_outer_iter_idx % self.local_num_iterations == 0)

    def check_test_frequency(self):
        return self.client_timer.local_outer_epoch_idx % self.args.frequency_of_the_test == 0 \
            or self.client_timer.local_outer_epoch_idx == self.args.max_epochs - 1

    def lr_schedule(self, num_iterations, warmup_epochs):

        # epochs = self.client_timer.local_outer_epoch_idx
        # iterations = self.client_timer.local_outer_iter_idx
        epochs = self.client_timer.global_outer_epoch_idx
        iterations = self.client_timer.global_outer_iter_idx
        if self.args.sched == "no":
            pass
        else:
            if epochs < warmup_epochs:
                self.trainer.warmup_lr_schedule(iterations)
            else:
                # When epoch begins, do lr_schedule.
                if (iterations > 0 and iterations % num_iterations == 0):
                    self.trainer.lr_schedule(epochs)


    def handle_message_init(self, msg_params):
        client_index, model_params, global_other_params, time_info, \
            global_train_tracker_info, global_test_tracker_info = self.process_model_message(msg_params)

        self.client_timer.update_time_info(time_info)
        self.client_timer.past_comm_round(comm_round=1)
        self.algorithm_on_handle_message_init(
            client_index, model_params, global_other_params, time_info,
            global_train_tracker_info, global_test_tracker_info
        )



    @abstractmethod
    def algorithm_on_handle_message_init(
        self, client_index, model_params, global_other_params, time_info,
        global_train_tracker_info, global_test_tracker_info
    ):
        pass



    def handle_message_receive_model_from_server(self, msg_params):
        logging.info("handle_message_receive_model_from_server.")
        client_index, model_params, global_other_params, time_info, \
            global_train_tracker_info, global_test_tracker_info = self.process_model_message(msg_params)
        self.client_timer.update_time_info(time_info)
        self.client_timer.past_comm_round(comm_round=1)
        self.algorithm_on_handle_message_receive_model_from_server(
            client_index, model_params, global_other_params, time_info,
            global_train_tracker_info, global_test_tracker_info
        )

    @abstractmethod
    def algorithm_on_handle_message_receive_model_from_server(
        self, client_index, model_params, global_other_params, time_info,
        global_train_tracker_info, global_test_tracker_info
    ):
        pass

    def handle_message_receive_grad_from_server(self, msg_params):
        logging.info("handle_message_receive_model_from_server.")
        client_index, grad_params, global_other_params, time_info, \
            global_train_tracker_info, global_test_tracker_info = self.process_grad_message(msg_params)
        self.client_timer.update_time_info(time_info)
        self.client_timer.past_comm_round(comm_round=1)
        self.algorithm_on_handle_message_receive_grad_from_server(
            client_index, grad_params, global_other_params, time_info,
            global_train_tracker_info, global_test_tracker_info
        )

    @abstractmethod
    def algorithm_on_handle_message_receive_grad_from_server(
        self, client_index, grad_params, global_other_params, time_info,
        global_train_tracker_info, global_test_tracker_info
    ):
        pass



    def bsp_style_train(self, client_index, named_params, params_type='model', global_other_params=None, traininig_start=False):

        if params_type == 'model':
            logging.debug(f"Client recv model............")
            self.trainer.set_model_params(named_params)
        elif params_type == 'grad':
            self.trainer.clear_grad_params()
            self.trainer.set_grad_params(named_params)
            self.trainer.update_model_with_grad()
            # if "averaged_bn_params" in global_other_params:
            #     self.trainer.trainer.set_model_bn(global_other_params["averaged_bn_params"])
            # self.trainer.trainer.set_model_bn(global_other_params["averaged_bn_params"])
            # raise NotImplementedError

        if traininig_start:
            self.trainer.update_dataset(int(client_index), self.client_timer.local_outer_epoch_idx)
            self.epoch_init()
        else:
            self.client_timer.past_iterations(iterations=1)
            if self.check_end_epoch():
                self.trainer.update_dataset(int(client_index), self.client_timer.local_outer_epoch_idx)
                self.epoch_init()
        self.lr_schedule(self.global_num_iterations, self.args.warmup_epochs)

        client_other_params = {}

        if self.args.psgd_exchange == 'model':
            weights_diff, model_indexes, local_sample_num = \
                self.trainer.train_one_step(
                    global_other_params,
                    self.client_timer.local_outer_epoch_idx, 
                    self.client_timer.local_inner_iter_idx,
                    self.check_end_epoch(),
                    self.train_tracker,
                    self.metrics
                )
            train_tracker_info = self.train_tracker.encode_local_info(
                client_index, if_reset=True, metrics=self.metrics)
            time_info = self.client_timer.get_time_info_to_send()
            self.send_model_to_server(
                0, weights_diff, local_sample_num, model_indexes,
                client_other_params=client_other_params,
                time_info=time_info,
                train_tracker_info=train_tracker_info,
                test_tracker_info=None)

        elif self.args.psgd_exchange == 'grad':
            grads, grad_indexes, local_sample_num = \
                self.trainer.infer_bw_one_step(
                    global_other_params,
                    self.client_timer.local_outer_epoch_idx, 
                    self.client_timer.local_inner_iter_idx,
                    self.check_end_epoch(),
                    self.train_tracker,
                    self.metrics
                )
            # TODO

            all_bn_params = self.trainer.trainer.get_model_bn()
            client_other_params["all_bn_params"] = all_bn_params
            train_tracker_info = self.train_tracker.encode_local_info(
                client_index, if_reset=True, metrics=self.metrics)
            time_info = self.client_timer.get_time_info_to_send()
            self.send_grad_to_server(
                0, grads, local_sample_num, grad_indexes,
                client_other_params=client_other_params,
                time_info=time_info,
                train_tracker_info=train_tracker_info,
                test_tracker_info=None)
        else:
            raise NotImplementedError


    def local_sgd_style_train(self, client_index, named_params, params_type='model', global_other_params=None, traininig_start=False):

        self.trainer.set_model_params(named_params)

        # use this for compression
        # Note here it is trainer.trainer
        if self.args.if_get_diff:
            previous_model = copy.deepcopy(self.trainer.trainer.get_model_params())
        for _ in range(self.local_round_num):

            self.client_timer.past_iterations(iterations=1)
            # do if branch here for update dataset
            # the new training round begin here.
            if self.check_end_epoch():
                self.trainer.update_dataset(int(client_index), self.client_timer.local_outer_epoch_idx)
                self.epoch_init()

            # Note here we input total_iteration into lr_schedule rather than round_idx.
            # self.lr_schedule(self.local_num_iterations, self.args.warmup_epochs)
            self.lr_schedule(self.global_num_iterations, self.args.warmup_epochs)

            # get model params out of the loop
            _, _, local_sample_num = \
                self.trainer.train_one_step(
                    global_other_params,
                    self.client_timer.local_outer_epoch_idx, 
                    self.client_timer.local_inner_iter_idx,
                    self.check_end_epoch(),
                    self.train_tracker,
                    self.metrics
                )

        client_other_params = {}

        if self.args.if_get_diff:
            weights, model_indexes = self.trainer.get_model_diff_params(previous_model)
        else:
            weights, model_indexes = self.trainer.get_model_params()

        train_tracker_info = self.train_tracker.encode_local_info(
            client_index, if_reset=True, metrics=self.metrics)

        time_info = self.client_timer.get_time_info_to_send()
        self.send_model_to_server(
            0, weights, local_sample_num, model_indexes,
            client_other_params=client_other_params,
            time_info=time_info,
            train_tracker_info=train_tracker_info,
            test_tracker_info=None)



    def client_local_test(self):
        if self.check_end_epoch():
            if self.check_test_frequency():
                self.trainer.local_test(
                    self.client_timer.local_outer_epoch_idx, self.test_tracker, self.metrics)
                # will be sent to server for uploading
                test_tracker_info = self.test_tracker.encode_local_info(
                    self.client_index, if_reset=True, metrics=self.metrics)
            else:
                test_tracker_info = None 
                self.test_tracker.reset()
        return test_tracker_info



    def get_tracker_info_from_message(self, msg_params):
        global_train_tracker_info = None 
        global_test_tracker_info = None
        # local_train_tracker_info = msg_params.get(MyMessage.MSG_ARG_KEY_TRAIN_TRACKER_INFO)
        # local_test_tracker_info = msg_params.get(MyMessage.MSG_ARG_KEY_TEST_TRACKER_INFO)

        # if local_train_tracker_info is not None:
        #     logging.debug('Server: receive train_tracker_info')
        #     # assert local_train_tracker_info['n_samples'] > 0
        #     self.total_train_tracker.decode_info(local_train_tracker_info)

        # if local_test_tracker_info is not None:
        #     logging.debug('Server: receive test_tracker_info')
        #     # assert local_test_tracker_info['n_samples'] > 0
        #     self.total_test_tracker.decode_info(local_test_tracker_info)
        return global_train_tracker_info, global_test_tracker_info


    def send_model_to_server(self, receive_id, weights, local_sample_num, model_indexes=None,
                            client_other_params=None,
                            time_info=None, train_tracker_info=None, test_tracker_info=None):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_INDEXES, model_indexes)

        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OTHER_PARAMS, client_other_params)

        message.add_params(MyMessage.MSG_ARG_KEY_TIME_INFO, time_info)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        # train
        message.add_params(MyMessage.MSG_ARG_KEY_TRAIN_TRACKER_INFO, train_tracker_info)
        # test
        message.add_params(MyMessage.MSG_ARG_KEY_TEST_TRACKER_INFO, test_tracker_info)
        self.send_message(message)

    def send_grad_to_server(self, receive_id, grads, local_sample_num, grad_indexes=None,
                            client_other_params=None,
                            time_info=None, train_tracker_info=None, test_tracker_info=None):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_GRAD_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_GRAD_PARAMS, grads)
        message.add_params(MyMessage.MSG_ARG_KEY_GRAD_INDEXES, grad_indexes)

        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OTHER_PARAMS, client_other_params)

        message.add_params(MyMessage.MSG_ARG_KEY_TIME_INFO, time_info)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        # train
        message.add_params(MyMessage.MSG_ARG_KEY_TRAIN_TRACKER_INFO, train_tracker_info)
        # test
        message.add_params(MyMessage.MSG_ARG_KEY_TEST_TRACKER_INFO, test_tracker_info)
        self.send_message(message)

    def process_grad_message(self, msg_params):
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        grad_params = msg_params.get(MyMessage.MSG_ARG_KEY_GRAD_PARAMS)
        global_other_params = msg_params.get(MyMessage.MSG_ARG_KEY_GLOBAL_OTHER_PARAMS)
        time_info = msg_params.get(MyMessage.MSG_ARG_KEY_TIME_INFO)
        self.client_timer.update_time_info(time_info)

        global_train_tracker_info, global_test_tracker_info = \
            self.get_tracker_info_from_message(msg_params)

        return client_index, grad_params, global_other_params, time_info, \
            global_train_tracker_info, global_test_tracker_info

    def process_model_message(self, msg_params):
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        global_other_params = msg_params.get(MyMessage.MSG_ARG_KEY_GLOBAL_OTHER_PARAMS)
        time_info = msg_params.get(MyMessage.MSG_ARG_KEY_TIME_INFO)
        self.client_timer.update_time_info(time_info)

        global_train_tracker_info, global_test_tracker_info = \
            self.get_tracker_info_from_message(msg_params)

        return client_index, model_params, global_other_params, time_info, \
            global_train_tracker_info, global_test_tracker_info




