import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))


# from fedml_core.distributed.client.client_manager import ClientManager
from fedml_core.distributed.communication.message import Message

from algorithms.basePS.ps_client_manager import PSClientManager


from utils.context import (
    raise_error_without_process,
    get_lock,
)

from .message_define import MyMessage

class PSGDClientManager(PSClientManager):
    def __init__(self, args, trainer, comm=None, rank=0, size=0, backend="MPI", perf_timer=None, metrics=None):
        super().__init__(args, trainer, comm, rank, size, backend, perf_timer, metrics)

    def get_max_comm_round(self):
        return self.args.max_epochs * self.global_num_iterations + 1


    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_INIT_CONFIG,
                                            self.handle_message_init)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
                                            self.handle_message_receive_model_from_server)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SYNC_GRAD_TO_CLIENT,
                                            self.handle_message_receive_grad_from_server)


    def algorithm_on_handle_message_init(
        self, client_index, model_params, global_other_params, time_info,
        global_train_tracker_info, global_test_tracker_info
    ):
        self.bsp_style_train(
            client_index, 
            model_params, 
            params_type='model',
            global_other_params=global_other_params, 
            traininig_start=True
        )


    def algorithm_on_handle_message_receive_model_from_server(
        self, client_index, model_params, global_other_params, time_info,
        global_train_tracker_info, global_test_tracker_info
    ):
        self.bsp_style_train(
            client_index, 
            model_params, 
            params_type='model',
            global_other_params=global_other_params, 
            traininig_start=False
        )


    """ In this implementation, server update model and then sends newest model to clients.
        for conveniently global testing. 
    """
    def algorithm_on_handle_message_receive_grad_from_server(
        self, client_index, grad_params, global_other_params, time_info,
        global_train_tracker_info, global_test_tracker_info
    ):
        # raise NotImplementedError
        self.bsp_style_train(
            client_index, 
            grad_params, 
            params_type='grad',
            global_other_params=global_other_params, 
            traininig_start=False
        )
