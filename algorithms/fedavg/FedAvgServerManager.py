import logging
import os
import sys
import numpy as np

from .message_define import MyMessage

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_core.distributed.communication.message import Message
# from fedml_core.distributed.server.server_manager import ServerManager

from algorithms.basePS.ps_server_manager import PSServerManager

from utils.timer import Timer
from utils.tracker import RuntimeTracker, get_metric_info
from utils.metrics import Metrics
from utils.wandb_util import wandb_log, upload_metric_info

from .message_define import MyMessage


class FedAVGServerManager(PSServerManager):
    def __init__(self, args, aggregator, comm=None, rank=0, size=0, backend="MPI", timer=None, metrics=None):
        super().__init__(args, aggregator, comm, rank, size, backend, timer, metrics)
        # assert args.client_num_in_total == self.size - 1
        assert args.client_num_per_round == self.size - 1

    def run(self):
        super().run()

    def get_comm_round(self):
        return self.args.comm_round + 1

    # @overrid
    def send_init_msg(self):
        # sampling clients
        client_indexes = self.aggregator.client_sampling(self.global_round_idx, self.args.client_num_in_total,
                                                         self.args.client_num_per_round)
        global_model_params = self.aggregator.get_global_model_params()
        for process_id in range(1, self.size):
            self.send_message_init_config(process_id, global_model_params, client_indexes[process_id - 1],
                                          global_round_idx=self.global_round_idx)

    # override
    def choose_clients_and_send(self, global_model_params, params_type='model', global_round_idx=None):
        # sampling clients
        client_indexes = self.aggregator.client_sampling(self.global_round_idx, self.args.client_num_in_total,
                                                            self.args.client_num_per_round)
        logging.debug("size = %d" % self.size)
        for receiver_id in range(1, self.size):
            if params_type == 'grad':
                self.send_message_sync_grad_to_client(receiver_id, global_model_params,
                                                    client_indexes[receiver_id - 1],
                                                    global_round_idx=global_round_idx)
            elif params_type == 'model':
                self.send_message_sync_model_to_client(receiver_id, global_model_params,
                                                        client_indexes[receiver_id - 1],
                                                        global_round_idx=global_round_idx)


    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
                                              self.handle_message_receive_model_from_client)


    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        model_indexes = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_INDEXES)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)

        # only receive train metric info
        local_train_metric_info, _ = self.get_metric_info_from_message(msg_params)

        self.aggregator.add_local_trained_result(sender_id - 1, model_params, model_indexes, local_sample_number)
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.debug("b_all_received = " + str(b_all_received))
        if b_all_received:
            if self.args.Failure_chance is not None and np.random.rand(1) < self.args.Failure_chance:
                logging.info("Communication Failure happens on worker: {}, Failure_chance: {}".format(
                    self.worker_index, self.args.Failure_chance))
                global_model_params = self.aggregator.get_global_model_params()
            else:
                global_model_params = self.aggregator.aggregate()

            # Test on server
            if self.global_round_idx % self.args.frequency_of_the_test == 0 or self.global_round_idx == self.args.comm_round - 1:
                self.aggregator.test_on_server_for_all_clients(self.epoch, self.total_test_tracker, self.metrics)
                train_metric_info, test_metric_info = get_metric_info(
                    self.total_train_tracker, self.total_test_tracker, time_stamp=self.epoch, if_reset=True,
                    metrics=self.metrics)
                com_values = {"epoch": train_metric_info['time_stamp'], "round": self.global_round_idx}
                upload_metric_info(str_pre="Epoch: {}, Server Total: ".format(train_metric_info['time_stamp']),
                                    train_metric_info=train_metric_info,
                                test_metric_info=test_metric_info, metrics=self.metrics,
                                comm_values=com_values)
            else:
                self.reset_train_test_tracker()

            # start the next round
            self.global_round_idx += 1
            self.epoch += self.args.epochs
            if self.global_round_idx == self.comm_round:
                self.finish()
                return

            self.choose_clients_and_send(global_model_params,
                                         params_type='model', global_round_idx=self.global_round_idx)

