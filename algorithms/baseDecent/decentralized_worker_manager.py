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

from mpi4py import MPI


from utils.context import (
    raise_MPI_error,
    raise_error_without_process,
    get_lock,
)

from utils.tracker import RuntimeTracker
from utils.metrics import Metrics

from timers.client_timer import ClientTimer
from timers.server_timer import ServerTimer



class BaseDecentralizedWorkerManager(ClientManager):
    def __init__(self, args, comm, rank, size, worker, topology_manager, model_trainer, perf_timer, metrics):
        super().__init__(args, comm, rank, size)
        self.client_index = rank
        self.worker = worker
        self.size = size

        self.model_trainer = model_trainer
        self.topology_manager = topology_manager
        self.gossip_info = self.topology_manager.topology[self.client_index]

        self.max_comm_round = self.get_max_comm_round()

        self.flag_client_finish_dict = dict()
        for client_index in range(self.size):
            self.flag_client_finish_dict[client_index] = False

        self.perf_timer = perf_timer
        # ================================================
        self.client_other_params_dict = {}
        # ================================================
        self.local_num_iterations = self.worker.local_num_iterations
        self.global_num_iterations = self.worker.global_num_iterations
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

        self.coodinator_upload_results_flag = False

        if self.client_index == 0:
            self.coodinator_thread = threading.Thread(
                name="coordinator", target=self.run_coordinator)
            self.server_timer = ServerTimer(
                self.args,
                self.global_num_iterations,
                local_num_iterations_dict=None
            )
            self.total_train_tracker = RuntimeTracker(
                mode='Train',
                things_to_metric=self.metrics.metric_names,
                timer=self.server_timer,
                args=args
            )
            self.total_test_tracker = RuntimeTracker(
                mode='Test',
                things_to_metric=self.metrics.metric_names,
                timer=self.server_timer,
                args=args
            )
        else:
            self.total_test_tracker = None
            self.total_train_tracker = None


        self.training_thread = threading.Thread(name='training', target=self.run_sync)

        # For coordinator notify event
        self.all_clients_finish_event = threading.Event()
        self.total_metric_lock = threading.Lock()
        self.start_epoch_event = threading.Event()


    def get_max_comm_round(self):
        return self.args.max_epochs * self.worker.global_num_iterations

    def epoch_init(self):
        self.worker.epoch_init()


    def check_end_epoch(self):
        return (self.client_timer.global_outer_iter_idx > 0 and \
            self.client_timer.global_outer_iter_idx % self.global_num_iterations == 0)

    def check_test_frequency(self):
        return self.client_timer.global_outer_epoch_idx % self.args.frequency_of_the_test == 0 \
            or self.client_timer.global_outer_epoch_idx == self.args.max_epochs - 1


    def client_check_whether_all_receive_and_process(self):
        if self.worker.check_whether_all_receive():
            logging.debug(">>>>>>>>>>>>>>>WORKER %d, ROUND %d finished!<<<<<<<<" % (self.client_index, self.client_timer.global_comm_round_idx))
            # self.client_timer.global_comm_round_idx += 1

            # not needed in run_async 
            self.sync_receive_all_event.set()

    def lr_schedule(self, num_iterations, warmup_epochs):
    
        epochs = self.client_timer.global_outer_epoch_idx
        iterations = self.client_timer.global_outer_iter_idx
        if self.args.sched == "no":
            pass
        else:
            if epochs < warmup_epochs:
                self.model_trainer.warmup_lr_schedule(iterations)
            else:
                # When epoch begins, do lr_schedule.
                if (iterations > 0 and iterations % num_iterations == 0):
                    self.model_trainer.lr_schedule(epochs)




    def handle_msg_client_to_coordinator(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        logging.debug("handle_msg_client_to_coordinator. sender_id = " + str(sender_id))
        client_index = sender_id 

        with get_lock(self.total_metric_lock):
            logging.debug("get metric lock, handle_msg_client_to_coordinator. sender_id = " + str(sender_id))
            self.flag_client_finish_dict[sender_id] = True

            client_other_params = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_OTHER_PARAMS)
            self.client_other_params_dict[sender_id] = client_other_params

            time_info = msg_params.get(MyMessage.MSG_ARG_KEY_TIME_INFO)
            self.server_timer.update_time_info(time_info)

            train_tracker_info = msg_params.get(MyMessage.MSG_ARG_KEY_TRAIN_TRACKER_INFO)
            if train_tracker_info is not None:
                self.total_train_tracker.decode_local_info(client_index, train_tracker_info)

            test_tracker_info = msg_params.get(MyMessage.MSG_ARG_KEY_TEST_TRACKER_INFO)
            if test_tracker_info is not None:
                self.coodinator_upload_results_flag = True
                self.total_test_tracker.decode_local_info(client_index, test_tracker_info)
            else:
                self.coodinator_upload_results_flag = False

            self.check_worker_finish_and_notify()


    def handle_msg_coordinator_to_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        global_other_params = msg_params.get(MyMessage.MSG_ARG_KEY_GLOBAL_OTHER_PARAMS)
        time_info = msg_params.get(MyMessage.MSG_ARG_KEY_TIME_INFO)
        self.client_timer.update_time_info(time_info)

        logging.debug("handle_msg_coordinator_to_client. sender_id = " + str(sender_id))
        self.start_epoch_event.set()


    def test_and_send_to_coordinator(self, client_other_params):
        if self.check_end_epoch() and self.check_test_frequency():
            # if iteration > 0:
            """ one epoch ends here"""
            # test
            self.worker.test(self.client_timer.global_outer_epoch_idx, self.test_tracker, self.metrics)

            train_tracker_info = self.train_tracker.encode_local_info(
                self.client_index, if_reset=True, metrics=self.metrics)
            test_tracker_info = self.test_tracker.encode_local_info(
                self.client_index, if_reset=True, metrics=self.metrics)

            if self.rank == 0:
                with get_lock(self.total_metric_lock):
                    self.flag_client_finish_dict[self.rank] = True
                    self.total_train_tracker.decode_local_info(0, test_tracker_info)
                    self.total_test_tracker.decode_local_info(0, train_tracker_info)
                    time_info = self.client_timer.get_time_info_to_send()
                    self.server_timer.update_time_info(time_info)
                    self.check_worker_finish_and_notify()
            else:
                self.send_notify_to_coordinator(0, client_other_params, train_tracker_info, test_tracker_info)
        else:
            self.train_tracker.reset()
            self.test_tracker.reset()
            # self.reset_train_test_tracker(self.train_tracker, self.test_tracker)
            if self.rank == 0:
                with get_lock(self.total_metric_lock):
                    self.flag_client_finish_dict[self.rank] = True
                    self.check_worker_finish_and_notify()
            else:
                self.send_notify_to_coordinator(0, client_other_params)



        # if self.check_end_epoch():
        #     if self.check_test_frequency():
        #         # if iteration > 0:
        #         """ one epoch ends here"""
        #         # test
        #         self.worker.test(self.client_timer.global_outer_epoch_idx, self.test_tracker, self.metrics)

        #         train_tracker_info = self.train_tracker.encode_local_info(
        #             self.client_index, if_reset=True, metrics=self.metrics)
        #         test_tracker_info = self.test_tracker.encode_local_info(
        #             self.client_index, if_reset=True, metrics=self.metrics)

        #         if self.rank == 0:
        #             with get_lock(self.total_metric_lock):
        #                 self.flag_client_finish_dict[self.rank] = True
        #                 self.total_train_tracker.decode_local_info(0, test_tracker_info)
        #                 self.total_test_tracker.decode_local_info(0, train_tracker_info)
        #                 time_info = self.client_timer.get_time_info_to_send()
        #                 self.server_timer.update_time_info(time_info)
        #                 self.check_worker_finish_and_notify()
        #         else:
        #             self.send_notify_to_coordinator(0, train_tracker_info, test_tracker_info)
        #     else:
        #         self.train_tracker.reset()
        #         self.test_tracker.reset()
        # else:
        #     # self.reset_train_test_tracker(self.train_tracker, self.test_tracker)
        #     if self.rank == 0:
        #         with get_lock(self.total_metric_lock):
        #             self.flag_client_finish_dict[self.rank] = True
        #             self.check_worker_finish_and_notify()
        #     else:
        #         self.send_notify_to_coordinator(0)




    def run_coordinator(self):
        with raise_MPI_error(MPI):
            for epoch in range(self.args.max_epochs):
                for iteration in range(self.worker.global_num_iterations):
                    self.all_clients_finish_event.wait()
                    if self.coodinator_upload_results_flag:
                        self.total_train_tracker.upload_metric_and_suammry_info_to_wandb(if_reset=True, metrics=self.metrics)
                        self.total_test_tracker.upload_metric_and_suammry_info_to_wandb(if_reset=True, metrics=self.metrics)
                    else:
                        pass
                    # self.server_timer.past_iterations(iterations=1)
                    self.server_timer.past_iterations(iterations=1)
                    self.server_timer.past_comm_round(comm_round=1)
                    self.all_clients_finish_event.clear()
            self.finish()

    def check_whether_all_clients_finish_receive(self):
        for rank, flag in self.flag_client_finish_dict.items():
            if not flag:
                return False
        for rank, _ in self.flag_client_finish_dict.items():
            self.flag_client_finish_dict[rank] = False
        return True


    def check_worker_finish_and_notify(self, global_other_params={}):
        if self.check_whether_all_clients_finish_receive():
            logging.debug(">>>>>>>>>>>>>>>COORDINATOR Receive all, ROUND %d finished!<<<<<<<<" %
                (self.server_timer.global_comm_round_idx))
            self.all_clients_finish_event.set()
            self.notify_clients(global_other_params)
            if self.server_timer.global_comm_round_idx == self.max_comm_round:
                self.finish()


    def notify_clients(self, global_other_params={}):
        logging.debug("COORDINATOR notify clients to start!")
        for client_index in range(self.size):
            if client_index == 0:
                self.start_epoch_event.set()
                time_info = self.server_timer.get_time_info_to_send()
                self.client_timer.update_time_info(time_info)
            else:
                self.send_notify_to_clients(client_index, global_other_params)


    def send_message_init_config(self, receive_id, global_other_params=None):
        message = Message(MyMessage.MSG_TYPE_INIT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_GLOBAL_OTHER_PARAMS, global_other_params)
        self.send_message(message)

    def send_notify_to_coordinator(self, receive_id=0, client_other_params=None, 
                                train_tracker_info=None, test_tracker_info=None):
        logging.debug("send_notify_to_coordinator. receive_id = %s, round: %s" % (
            str(receive_id), str(self.client_timer.global_comm_round_idx)))
        message = Message(MyMessage.MSG_TYPE_CLIENT_TO_COORDINATOR, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OTHER_PARAMS, client_other_params)

        # time info 
        time_info = self.client_timer.get_time_info_to_send()
        message.add_params(MyMessage.MSG_ARG_KEY_TIME_INFO, time_info)

        # train
        message.add_params(MyMessage.MSG_ARG_KEY_TRAIN_TRACKER_INFO, train_tracker_info)

        # test
        message.add_params(MyMessage.MSG_ARG_KEY_TEST_TRACKER_INFO, test_tracker_info)
        self.send_message(message)

    def send_notify_to_clients(self, receive_id, global_other_params=None):
        logging.debug("send_notify_to_clients. receive_id = %s, round: %s" % (
            str(receive_id), str(self.server_timer.global_comm_round_idx)))

        message = Message(MyMessage.MSG_TYPE_COORDINATOR_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_GLOBAL_OTHER_PARAMS, global_other_params)
        # time info 
        time_info = self.server_timer.get_time_info_to_send()
        message.add_params(MyMessage.MSG_ARG_KEY_TIME_INFO, time_info)

        self.send_message(message)

    def send_result_to_neighbors(self, receive_id, client_params1, local_sample_number, client_other_params=None):
        logging.debug("send_result_to_neighbors. receive_id = " + str(receive_id))
        message = Message(MyMessage.MSG_TYPE_SEND_MSG_TO_NEIGHBOR, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, client_params1)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_number)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OTHER_PARAMS, client_other_params)
        self.send_message(message)

    def send_sparse_params_to_neighbors(self, receive_id, client_sparse_params1, client_sparse_index1, local_sample_number, client_other_params):
        logging.debug("send_sparse_params_to_neighbors. receive_id = " + str(receive_id))
        message = Message(MyMessage.MSG_TYPE_SEND_MSG_TO_NEIGHBOR, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, client_sparse_params1)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_INDEXES, client_sparse_index1)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_number)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OTHER_PARAMS, client_other_params)
        self.send_message(message)

    def send_quant_params_to_neighbors(self, receive_id, client_quant_params1, local_sample_number, client_other_params):
        logging.debug("send_quant_params_to_neighbors. receive_id = " + str(receive_id))
        message = Message(MyMessage.MSG_TYPE_SEND_MSG_TO_NEIGHBOR, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, client_quant_params1)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_number)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OTHER_PARAMS, client_other_params)
        self.send_message(message)

    def send_sign_params_to_neighbors(self, receive_id, client_sign_params1, local_sample_number, client_other_params):
        logging.debug("send_sign_params_to_neighbors. receive_id = " + str(receive_id))
        message = Message(MyMessage.MSG_TYPE_SEND_MSG_TO_NEIGHBOR, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, client_sign_params1)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_number)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OTHER_PARAMS, client_other_params)
        self.send_message(message)


