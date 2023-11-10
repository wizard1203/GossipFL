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

from .compressor import SAPS_FLCompressor
from .utils import generate_bandwidth

from .SAPS_topology_manager import SAPSTopologyManager
from .message_define import MyMessage



comm = MPI.COMM_WORLD

class DecentralizedWorkerManager(BaseDecentralizedWorkerManager):
    def __init__(self, args, comm, rank, size, worker, topology_manager, model_trainer, timer, metrics):
        super().__init__(args, comm, rank, size, worker, topology_manager, model_trainer, timer, metrics)
        # self.topology_manager.generate_topology(t=self.global_round_idx)
        # self.training_thread = threading.Thread(name='training', target=self.run_async)
        # self.training_thread.start()
        self.neighbor_transfer_lock = threading.Lock()
        self.sync_receive_all_event = threading.Event()
        self.complete_aggregation_event = threading.Event()

        # compression part
        self.compression = args.compression
        assert self.compression in ["topk", "randomk", "quantize", "sign"]
        self.compressor = SAPS_FLCompressor(comm_op=self.compression,
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


    def handle_msg_from_neighbor(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        # training_interation_result = msg_params.get(MyMessage.MSG_ARG_KEY_PARAMS_1)
        # local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)

        # =========================== wait for complete aggregation
        logging.debug("receive rank %d message, wait for complete aggregation" %
            (sender_id))

        with get_lock(self.neighbor_transfer_lock):
            logging.debug("handle_msg_from_neighbor. sender_id = " + str(sender_id))
            # self.worker.add_result(sender_id, training_interation_result)
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

        logging.debug("receive rank %d message, msg_params.get(MyMessage.MSG_ARG_KEY_SPARSE_INDEX_1): %s" % (
             sender_id, msg_params.get(MyMessage.MSG_ARG_KEY_SPARSE_INDEX_1)[0:10] )
        )

        self.worker.flag_neighbor_result_received_dict[sender_id] = True
        assert self.worker.shapes is not None
        self.compressor.original_shapes = self.worker.shapes
        if self.args.Failure_chance is not None and np.random.rand(1) < self.args.Failure_chance:
            logging.info("Communication Failure happens on worker: {}, Failure_chance: {}".format(
                self.worker_index, self.args.Failure_chance))
        else:
            self.compressor.uncompress(msg_params, self.gossip_info[sender_id],
                self.worker.neighbor_hat_params["memory"], 
                self.selected_shapes, self.worker.shapes,
                self.worker.device
            )

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
            # for the first iteration usage
            self.topology_manager.generate_topology(t=self.global_round_idx)
            self.worker.refresh_gossip_info()
            self.refresh_gossip_info()
            # reset the neighbor_hat_params for storing new values
            self.worker.init_neighbor_hat_params()
            for epoch in range(self.epochs):
                self.epoch = epoch

                # update worker's dataset and data loaderw
                with raise_error_without_process():
                    self.worker.train_local.sampler.set_epoch(epoch)
                # self.worker.update_dataset()
                self.epoch_init()

                for iteration in range(self.worker.num_iterations):
                    self.iteration = iteration
                    logging.debug("wait start_epoch_event")

                    self.start_epoch_event.wait()
                    logging.debug("Begin iteration")

                    # Get model params
                    params, _ = get_data(
                        self.worker.param_groups, self.worker.param_names, is_get_grad=False
                    )
                    flatten_params = TensorBuffer(params)

                    # compress
                    sync_buffer = {
                        "original_shapes": self.worker.shapes,
                        "flatten_params": flatten_params,
                    }
                    self.compressor.compress(sync_buffer)
                    self.selected_shapes = sync_buffer["selected_shapes"]

                    # allow receive others' models at the beginning for acceleration
                    # Must after the selected_shapes 
                    # self.complete_aggregation_event.set()

                    # begin to send model
                    logging.debug("Begin send and receive")
                    logging.debug(self.topology_manager.topology)
                    for neighbor_idx in self.topology_manager.get_out_neighbor_idx_list(self.worker_index):
                        if self.compression in ["randomk", "topk"]:
                            self.send_sparse_params_to_neighbors(neighbor_idx, 
                                sync_buffer["flatten_selected_values"].buffer.cpu(),
                                sync_buffer["flatten_selected_indices"].buffer.cpu(),
                                self.worker.get_dataset_len())
                        # Not support Now
                        elif self.compression == "quantize":
                            raise NotImplementedError
                        # Not support Now
                        elif self.compression == "sign":
                            raise NotImplementedError
                        else:
                            raise NotImplementedError
                    # wait for receiving all
                    self.sync_receive_all_event.wait()
                    self.worker.aggregate(self.compressor, self.selected_shapes, self.gossip_info)

                    # ready for aggregation in next epoch
                    # self.complete_aggregation_event.clear()

                    # Get weighted hat params and apply the local gradient.
                    self.neighbor_transfer_lock.acquire()


                    # add the sparsed part
                    self.compressor.uncompress_direct(
                        sync_buffer, self.worker.neighbor_hat_params["memory"],
                        self.selected_shapes, self.worker.shapes)
                    sync_buffer["flatten_params"].unpack(params)

                    if self.neighbor_transfer_lock.locked():
                        self.neighbor_transfer_lock.release()

                    if self.args.Failure_chance is not None and np.random.rand(1) < self.args.Failure_chance:
                        logging.info("Communication Failure happens on worker: {}, Failure_chance: {}".format(
                            self.worker_index, self.args.Failure_chance))
                    else:
                        self.lr_schedule(self.epoch, self.iteration, self.global_round_idx,
                                        self.worker.num_iterations, self.args.warmup_epochs)
                        # update x_half to x_{t+1} by SGD
                        loss, output, target \
                            = self.worker.train_one_step(self.epoch, self.iteration,
                                                            self.train_tracker, self.metrics)

                    # batch_metric_stat = self.metrics.evaluate(loss, output, target)
                    # self.train_tracker.update_metrics(batch_metric_stat, n_samples=target.size(0))
                    # logging.info('Local Training Epoch: {} iter: {} \t Loss: {:.6f}, Acc1: {:.6f}'.format(
                    #                 epoch, iteration, batch_metric_stat['Loss'], batch_metric_stat['Acc1']))
                    # logging.info('(Local Training Epoch: {}, Iter: {} '.format(
                    #     epoch, iteration) + self.metrics.str_fn(batch_metric_stat))
                    self.start_epoch_event.clear()
                    self.sync_receive_all_event.clear()

                    """
                        Before send msg to coordinator,
                        generate topology firstly.
                    """
                    self.topology_manager.generate_topology(t=self.global_round_idx)
                    self.worker.refresh_gossip_info()
                    self.refresh_gossip_info()
                    # reset the neighbor_hat_params for storing new values
                    self.worker.init_neighbor_hat_params()

                    self.test_and_send_to_coordinator(iteration, epoch)
                # self.model_trainer.lr_schedule(epoch+1)



    def refresh_gossip_info(self):
        self.neighbors_info = self.topology_manager.topology
        self.gossip_info = self.topology_manager.topology[self.worker_index]


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

    def send_sign_params_to_neighbors(self, receive_id, client_sign_params1, local_sample_number):
        logging.debug("send_sign_params_to_neighbors. receive_id = " + str(receive_id))
        message = Message(MyMessage.MSG_TYPE_SEND_MSG_TO_NEIGHBOR, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_SIGN_PARAMS_1, client_sign_params1)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_number)
        self.send_message(message)




