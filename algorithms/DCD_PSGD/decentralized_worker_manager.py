import logging
import threading
import time
from copy import deepcopy

import traceback
from mpi4py import MPI
import numpy as np

from fedml_core.distributed.client.client_manager import ClientManager
from fedml_core.distributed.communication.message import Message

from algorithms.baseDecent.decentralized_worker_manager import BaseDecentralizedWorkerManager

from utils.context import (
    raise_MPI_error,
    raise_error_without_process,
    get_lock,
)
from utils.timer import Timer
from utils.tracker import RuntimeTracker
# from fedml_api.utils.timer_with_cuda import Timer
from utils.metrics import Metrics
from utils.wandb_util import wandb_log
from utils.data_utils import (
    get_data,
    apply_gradient
)
from utils.tensor_buffer import (
    TensorBuffer
)


from .message_define import MyMessage
from .compressor import DCDCompressor


comm = MPI.COMM_WORLD

class DecentralizedWorkerManager(BaseDecentralizedWorkerManager):
    def __init__(self, args, comm, rank, size, worker, topology_manager, model_trainer, timer, metrics):
        super().__init__(args, comm, rank, size, worker, topology_manager, model_trainer, timer, metrics)
        # ====================================
        self.training_thread = threading.Thread(name='training', target=self.run_sync)

        # self.training_thread = threading.Thread(name='training', target=self.run_async)
        # self.training_thread.start()
        self.model_transfer_lock = threading.Lock()
        self.neighbor_transfer_lock = threading.Lock()
        self.sync_receive_all_event = threading.Event()
        self.complete_aggregation_event = threading.Event()


        # compression part
        self.compression = args.compression
        assert self.compression in ["topk", "randomk", "quantize", "sign"]
        self.compressor = DCDCompressor(comm_op=self.compression,
                                        compress_ratio=args.compress_ratio,
                                        quantize_level=args.quantize_level,
                                        is_biased=(args.is_biased == 1),)

    def run(self):
        # self.start_training()
        self.training_thread.start()
        logging.debug("Wait for the barrier!")
        comm.Barrier()
        time.sleep(1)
        logging.debug("MPI exit barrier!")

        if self.worker_index == 0:
            logging.debug("COORDINATOR notify clients to start!")
            self.coodinator_thread.start()
            self.notify_clients()
        super().run()


    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_SEND_MSG_TO_NEIGHBOR,
                                              self.handle_msg_from_neighbor)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_CLIENT_TO_COORDINATOR,
                                              self.handle_msg_client_to_coordinator)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_COORDINATOR_TO_CLIENT,
                                              self.handle_msg_coordinator_to_client)



    # def start_training(self):
    #     self.global_round_idx = 0
    #     self.__train()

    # TODO
    def handle_msg_from_neighbor(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        # training_interation_result = msg_params.get(MyMessage.MSG_ARG_KEY_PARAMS_1)
        # local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)

        # =========================== wait for complete aggregation
        logging.debug("receive rank %d message, wait for complete aggregation" %
            (sender_id))
        with get_lock(self.neighbor_transfer_lock):

            logging.debug("handle_msg_from_neighbor. sender_id = " + str(sender_id))
            self.worker.add_result(sender_id, msg_params)

        if self.worker.check_whether_all_receive():
            logging.debug(">>>>>>>>>>>>>>>WORKER %d, ROUND %d finished!<<<<<<<<" % (self.worker_index, self.global_round_idx))
            self.global_round_idx += 1

            self.sync_receive_all_event.set()
            if self.global_round_idx == self.comm_round:
                self.finish()
            # self.__train()

    def handle_msg_from_neighbor_old(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        # training_interation_result = msg_params.get(MyMessage.MSG_ARG_KEY_PARAMS_1)
        # local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)

        # =========================== wait for complete aggregation
        logging.debug("receive rank %d message, wait for complete aggregation" %
            (sender_id))
        self.complete_aggregation_event.wait()
        self.neighbor_transfer_lock.acquire()

        logging.debug("handle_msg_from_neighbor. sender_id = " + str(sender_id))
        # Not used to add compressed result into cache
        # self.worker.add_result(sender_id, training_interation_result)
        self.worker.flag_neighbor_result_received_dict[sender_id] = True
        assert self.worker.shapes is not None
        self.compressor.original_shapes = self.worker.shapes
        self.compressor.uncompress(msg_params, self.worker.neighbor_hat_params[sender_id], 
            self.selected_shapes, self.worker.shapes)
        if self.neighbor_transfer_lock.locked():
            self.neighbor_transfer_lock.release()
        if self.worker.check_whether_all_receive():
            logging.debug(">>>>>>>>>>>>>>>WORKER %d, ROUND %d finished!<<<<<<<<" % (self.worker_index, self.global_round_idx))
            self.global_round_idx += 1

            self.sync_receive_all_event.set()
            if self.global_round_idx == self.comm_round:
                self.finish()
            # self.__train()


    def run_sync(self):
        with raise_MPI_error():
            for epoch in range(self.epochs):
                self.epoch = epoch

                # update worker's dataset and data loader
                with raise_error_without_process():
                    self.worker.train_local.sampler.set_epoch(epoch)
                # self.worker.update_dataset()
                self.epoch_init()

                for iteration in range(self.worker.num_iterations):
                    self.iteration = iteration
                    logging.debug("wait start_epoch_event")
                    # self.start_epoch_event.set()
                    self.start_epoch_event.wait()
                    logging.debug("Begin iteration")

                    # robust test
                    if self.args.Failure_chance is not None and np.random.rand(1) < self.args.Failure_chance:
                        logging.info("Communication Failure happens on worker: {}, Failure_chance: {}".format(
                            self.worker_index, self.args.Failure_chance))
                    else:
                        self.lr_schedule(self.epoch, self.iteration, self.global_round_idx,
                                        self.worker.num_iterations, self.args.warmup_epochs)
                        loss, output, target \
                            = self.worker.infer_bw_one_step(self.epoch, self.iteration,
                                                            self.train_tracker, self.metrics)
                    local_sample_number = self.worker.local_sample_number
                    # batch_metric_stat = self.metrics.evaluate(loss, output, target)
                    # self.train_tracker.update_metrics(batch_metric_stat, n_samples=target.size(0))

                    # Apply the gradients with the weight decay and momentum.
                    logging.debug("Begin apply gradients")
                    apply_gradient(self.worker.param_groups, 
                        self.worker.model_trainer.get_optim_state(), apply_grad_to_model=False)

                    # Get model params and new grads
                    params, _ = get_data(
                        self.worker.param_groups, self.worker.param_names, is_get_grad=False
                    )
                    flatten_params = TensorBuffer(params)

                    grads, _ = get_data(
                        self.worker.param_groups, self.worker.param_names, is_get_grad=True
                    )
                    flatten_grads = TensorBuffer(grads)


                    # Get weighted hat params and apply the local gradient.
                    self.neighbor_transfer_lock.acquire()
                    # logging.info("self.worker.neighbor_hat_params.items()[1][1].device: {} \
                    #     , flatten_params.buffer.device: {}, flatten_grads.buffer.device: {} \
                    #     ".format(
                    #     list(self.worker.neighbor_hat_params.values())[0].buffer.device,
                    #     flatten_params.buffer.device,
                    #     flatten_grads.buffer.device
                    # ))
                    flatten_half_params = deepcopy(flatten_params)
                    flatten_half_params.buffer = (
                        sum(
                            [
                                _hat_params.buffer * self.gossip_info[_rank]
                                for _rank, _hat_params in self.worker.neighbor_hat_params.items()
                            ]
                        ) + (
                            flatten_params.buffer * self.gossip_info[self.worker_index]
                        )
                        - self.worker.param_groups[0]["lr"] * flatten_grads.buffer
                    )
                    if self.neighbor_transfer_lock.locked():
                        self.neighbor_transfer_lock.release()
                    # compress
                    # 
                    sync_buffer = {
                        "original_shapes": self.worker.shapes,
                        "flatten_half_params": flatten_half_params,
                        "flatten_params": flatten_params,
                    }
                    self.compressor.compress(sync_buffer)
                    self.selected_shapes = sync_buffer["selected_shapes"]

                    # has completed aggregation and compression, allow receive others' models 
                    # self.complete_aggregation_event.set()

                    # begin to send model
                    logging.debug("Begin send and receive")
                    for neighbor_idx in self.topology_manager.get_out_neighbor_idx_list(self.worker_index):
                        if self.compression in ["randomk", "topk"]:
                            self.send_sparse_params_to_neighbors(neighbor_idx, 
                                sync_buffer["flatten_selected_values"].buffer,
                                sync_buffer["flatten_selected_indices"].buffer,
                                local_sample_number)
                        # Not support Now
                        elif self.compression == "quantize":
                            self.send_quant_params_to_neighbors(neighbor_idx, 
                                sync_buffer["flatten_updates"].buffer,
                                local_sample_number)
                        # Not support Now
                        elif self.compression == "sign":
                            self.send_sigh_params_to_neighbors(neighbor_idx, 
                                sync_buffer["flatten_norms"].buffer,
                                local_sample_number)
                        else:
                            raise NotImplementedError
                    # wait for receiving all
                    self.sync_receive_all_event.wait()
                    self.worker.aggregate(self.compressor, self.selected_shapes)

                    # ready for aggregation in next epoch
                    # self.complete_aggregation_event.clear()

                    # update neighbor models
                    logging.debug("Begin uncompression of self-params")
                    assert self.worker.shapes is not None
                    self.compressor.uncompress_direct(
                        sync_buffer, flatten_params, self.selected_shapes, self.worker.shapes)

                    flatten_params.unpack(params)

                    # logging.info('Local Training Epoch: {} iter: {} \t Loss: {:.6f}, Acc1: {:.6f}'.format(
                    #                 epoch, iteration, batch_metric_stat['Loss'], batch_metric_stat['Acc1']))

                    self.start_epoch_event.clear()
                    self.sync_receive_all_event.clear()

                    self.test_and_send_to_coordinator(iteration, epoch)
                # self.model_trainer.lr_schedule(epoch+1)


    def send_result_to_neighbors(self, receive_id, client_params1, local_sample_number):
        logging.debug("send_result_to_neighbors. receive_id = " + str(receive_id))
        message = Message(MyMessage.MSG_TYPE_SEND_MSG_TO_NEIGHBOR, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_PARAMS_1, client_params1)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_number)
        self.send_message(message)

    def send_sparse_params_to_neighbors(self, receive_id, client_sparse_params1, client_sparse_index1, local_sample_number):
        logging.debug("send_sparse_params_to_neighbors. receive_id = " + str(receive_id))
        message = Message(MyMessage.MSG_TYPE_SEND_MSG_TO_NEIGHBOR, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_SPARSE_PARAMS_1, client_sparse_params1)
        message.add_params(MyMessage.MSG_ARG_KEY_SPARSE_INDEX_1, client_sparse_index1)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_number)
        self.send_message(message)

    def send_quant_params_to_neighbors(self, receive_id, client_quant_params1, local_sample_number):
        logging.debug("send_quant_params_to_neighbors. receive_id = " + str(receive_id))
        message = Message(MyMessage.MSG_TYPE_SEND_MSG_TO_NEIGHBOR, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_QUANT_PARAMS_1, client_quant_params1)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_number)
        self.send_message(message)

    def send_sigh_params_to_neighbors(self, receive_id, client_sign_params1, local_sample_number):
        logging.debug("send_sigh_params_to_neighbors. receive_id = " + str(receive_id))
        message = Message(MyMessage.MSG_TYPE_SEND_MSG_TO_NEIGHBOR, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_SIGN_PARAMS_1, client_sign_params1)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_number)
        self.send_message(message)


