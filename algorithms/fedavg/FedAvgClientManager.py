import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

# from fedml_core.distributed.client.client_manager import ClientManager
from fedml_core.distributed.communication.message import Message

from algorithms.basePS.ps_client_manager import PSClientManager


from .message_define import MyMessage


class FedAVGClientManager(PSClientManager):
    def __init__(self, args, trainer, comm=None, rank=0, size=0, backend="MPI", timer=None, metrics=None):
        super().__init__(args, trainer, comm, rank, size, backend, timer, metrics)


    def run(self):
        super().run()

    def get_comm_round(self):
        return self.args.comm_round + 1


    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_INIT_CONFIG,
                                              self.handle_message_init)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
                                              self.handle_message_receive_model_from_server)

    def update_time(self, global_round_idx):
        """
            Remember to revise them if needed.
        """
        self.global_round_idx = global_round_idx
        self.local_round_idx += 1
        self.epoch = self.local_round_idx * self.args.epochs
        self.iteration = 0
        self.total_iteration = self.epoch * self.local_num_iterations + self.iteration


    def handle_message_init(self, msg_params):
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        global_round_idx = msg_params.get(MyMessage.MSG_ARG_KEY_GLOBAL_ROUND_INDEX)

        logging.info("global_round_idx: {}".format(global_round_idx))
        self.trainer.set_model_params(global_model_params)
        train_metric_info = None
        test_metric_info = None

        self.update_time(global_round_idx)
        # the new training round begin here.
        self.trainer.update_dataset(int(client_index), self.epoch)

        weights_diff, model_indexes, local_sample_num = \
            self.trainer.fedavg_train(self.global_round_idx, self.epoch, self.iteration,
                                      self.train_tracker, self.metrics)

        # will be sent to server for uploading
        _, train_metric_info = self.get_metric_info(time_stamp=self.epoch, if_reset=True)

        self.send_model_to_server(0, weights_diff, local_sample_num, model_indexes,
                                global_round_idx=self.global_round_idx, local_round_idx=self.local_round_idx,
                                local_epoch_idx=self.epoch, local_iter_idx=self.iteration,
                                local_total_iter_idx=self.total_iteration)



    def handle_message_receive_model_from_server(self, msg_params):
        logging.debug("handle_message_receive_model_from_server.")
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        global_round_idx = msg_params.get(MyMessage.MSG_ARG_KEY_GLOBAL_ROUND_INDEX)

        self.trainer.set_model_params(model_params)
        train_metric_info = None
        test_metric_info = None

        self.update_time(global_round_idx)
        self.trainer.update_dataset(int(client_index), self.epoch)

        weights_diff, model_indexes, local_sample_num = \
            self.trainer.fedavg_train(self.global_round_idx, self.epoch, self.iteration,
                                      self.train_tracker, self.metrics)

        # will be sent to server for uploading
        _, train_metric_info = self.get_metric_info(time_stamp=self.epoch, if_reset=True)

        self.send_model_to_server(0, weights_diff, local_sample_num, model_indexes,
                                global_round_idx=self.global_round_idx, local_round_idx=self.local_round_idx,
                                local_epoch_idx=self.epoch, local_iter_idx=self.iteration,
                                local_total_iter_idx=self.total_iteration,
                                train_metric_info=train_metric_info)

