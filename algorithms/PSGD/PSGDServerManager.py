import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_core.distributed.communication.message import Message
# from fedml_core.distributed.server.server_manager import ServerManager

from algorithms.basePS.ps_server_manager import PSServerManager

from .message_define import MyMessage


class PSGDServerManager(PSServerManager):
    def __init__(self, args, aggregator, comm=None, rank=0, size=0, backend="MPI", perf_timer=None, metrics=None):
        super().__init__(args, aggregator, comm, rank, size, backend, perf_timer, metrics)
        assert args.client_num_in_total == self.size - 1
        assert args.client_num_per_round == self.size - 1

    def run(self):
        super().run()

    def get_max_comm_round(self):
        return self.args.max_epochs * self.global_num_iterations + 1

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
                                            self.handle_message_receive_model_from_client)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_GRAD_TO_SERVER,
                                            self.handle_message_receive_grad_from_client)

    def algorithm_on_handle_message_receive_model_from_client(
            self, sender_id, client_index, model_params, model_indexes, local_sample_number, client_other_params, time_info,
            local_train_tracker_info, local_test_tracker_info
        ):

        self.aggregator.add_local_trained_result(sender_id - 1, model_params, model_indexes, local_sample_number,
                                                client_other_params=client_other_params)
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.debug("b_all_received = " + str(b_all_received))
        if b_all_received:
            if self.args.Failure_chance is not None and np.random.rand(1) < self.args.Failure_chance:
                logging.info("Communication Failure happens on worker: {}, Failure_chance: {}".format(
                    self.client_index, self.args.Failure_chance))
                global_model_params = self.aggregator.get_global_model_params()
            else:
                global_model_params, global_other_params, shared_params_for_simulation = self.aggregator.aggregate(
                    global_comm_round=self.server_timer.global_comm_round_idx,
                    global_outer_epoch_idx=self.server_timer.global_outer_epoch_idx)

            self.check_and_test()
            # start the next round
            self.server_timer.past_iterations(iterations=1)
            self.server_timer.past_comm_round(comm_round=1)
            self.check_end_training()

            self.choose_clients_and_send(
                global_model_params, params_type='model', global_other_params=global_other_params)


    def algorithm_on_handle_message_receive_grad_from_client(
            self, sender_id, client_index, grad_params, grad_indexes, local_sample_number, client_other_params, time_info,
            local_train_tracker_info, local_test_tracker_info
        ):

        self.aggregator.add_local_trained_grad(client_index, grad_params, grad_indexes, local_sample_number,
                                            client_other_params=client_other_params)
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.debug("b_all_received = " + str(b_all_received))
        if b_all_received:
            grad_list, client_other_params_list, sample_num_list, training_num = \
                self.aggregator.get_received_params_list(params_type='grad')
            if self.args.Failure_chance is not None and np.random.rand(1) < self.args.Failure_chance:
                logging.debug("Communication Failure happens on worker: {}, Failure_chance: {}".format(
                    client_index, self.args.Failure_chance))
                global_grad_params = self.aggregator.aggregate_grads(
                    grad_list, client_other_params_list, sample_num_list, training_num,
                    global_comm_round=self.server_timer.global_comm_round_idx,
                    global_outer_epoch_idx=self.server_timer.global_outer_epoch_idx)
                for k in global_grad_params.keys():
                    global_grad_params[k] = global_grad_params[k].to(self.device) * 0
            else:
                global_grad_params = self.aggregator.aggregate_grads(
                    grad_list, client_other_params_list, sample_num_list, training_num,
                    global_comm_round=self.server_timer.global_comm_round_idx,
                    global_outer_epoch_idx=self.server_timer.global_outer_epoch_idx)

            # Remember to keep lr sche with clients!
            # Remember to clear grads of server optimizer!
            self.aggregator.clear_grad_params()
            self.lr_schedule(self.global_num_iterations, self.args.warmup_epochs)
            self.optimize_with_grad(global_grad_params)
            # global_model_params = self.aggregator.get_global_model_params()
            # self.aggregator.set_global_model_params(global_model_params)
            # self.aggregator.trainer.set_model_params(global_model_params)

            # all_bn_params = self.trainer.trainer.get_model_bn()
            # all_bn_params = client_other_params["all_bn_params"]
            averaged_bn_params = self.aggregator.aggregate_bn_params(
                client_other_params_list, sample_num_list, training_num,
                global_comm_round=self.server_timer.global_comm_round_idx,
                global_outer_epoch_idx=self.server_timer.global_outer_epoch_idx)
            self.aggregator.trainer.set_model_bn(averaged_bn_params)

            global_other_params = {}
            global_other_params["averaged_bn_params"] = averaged_bn_params
            global_model_params = self.aggregator.get_global_model_params()


            self.check_and_test()
            # start the next round
            self.server_timer.past_iterations(iterations=1)
            self.server_timer.past_comm_round(comm_round=1)
            self.check_end_training()

            self.choose_clients_and_send(
                global_grad_params, params_type='grad', global_other_params=global_other_params)
            # self.choose_clients_and_send(
            #     global_model_params, params_type='model', global_other_params=global_other_params)

