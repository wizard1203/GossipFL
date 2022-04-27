import logging
import threading
import time
from copy import deepcopy

import numpy as np
import torch
import traceback
from mpi4py import MPI

from fedml_core.distributed.client.client_manager import ClientManager
from fedml_core.distributed.communication.message import Message

from algorithms.baseDecent.decentralized_worker_manager import BaseDecentralizedWorkerManager

from .message_define import MyMessage

from mpi4py import MPI


from utils.context import (
    raise_MPI_error,
    raise_error_without_process,
    get_lock,
)


comm = MPI.COMM_WORLD

class DecentralizedWorkerManager(BaseDecentralizedWorkerManager):
    def __init__(self, args, comm, rank, size, worker, topology_manager, model_trainer, perf_timer, metrics):
        super().__init__(args, comm, rank, size, worker, topology_manager, model_trainer, perf_timer, metrics)
        # ====================================
        self.training_thread = threading.Thread(name='training', target=self.run_sync)

        # self.training_thread = threading.Thread(name='training', target=self.run_async)
        # self.training_thread.start()
        self.model_transfer_lock = threading.Lock()
        self.neighbor_transfer_lock = threading.Lock()
        self.sync_receive_all_event = threading.Event()
        self.complete_aggregation_event = threading.Event()


    def run(self):
        # self.start_training()
        self.training_thread.start()
        logging.debug("Wait for the barrier!")
        comm.Barrier()
        time.sleep(1)
        logging.debug("MPI exit barrier!")

        if self.client_index == 0:
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


    def handle_msg_from_neighbor(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        training_interation_result = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
        client_other_params = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_OTHER_PARAMS)

        # =========================== wait for complete aggregation
        logging.debug("Rank %d receive Rank %d message, wait for complete aggregation" %
            (self.rank, sender_id))
        # self.complete_aggregation_event.wait()

        # lock is actually not used in run_sync
        self.neighbor_transfer_lock.acquire()
        logging.debug("handle_msg_from_neighbor. sender_id = " + str(sender_id))
        self.worker.add_result(sender_id, training_interation_result)
        if self.neighbor_transfer_lock.locked():
            self.neighbor_transfer_lock.release()
        self.client_check_whether_all_receive_and_process()



    def run_sync(self):
        with raise_MPI_error(MPI):
            for _ in range(self.args.max_epochs):

                # update worker's dataset and data loader
                with raise_error_without_process():
                    self.worker.train_local.sampler.set_epoch(self.client_timer.local_outer_epoch_idx)
                # self.worker.update_dataset()
                self.epoch_init()

                for _ in range(self.worker.global_num_iterations):

                    logging.debug("wait start_epoch_event")
                    # self.start_epoch_event.set()
                    self.start_epoch_event.wait()

                    self.lr_schedule(self.worker.global_num_iterations, self.args.warmup_epochs)
                    if self.args.Failure_chance is not None and np.random.rand(1) < self.args.Failure_chance:
                        logging.info("Communication Failure happens on worker: {}, Failure_chance: {}".format(
                            self.client_index, self.args.Failure_chance))
                        # aggregated_model_params = self.worker.get_model_params()
                    else:
                        loss, output, target \
                            = self.worker.train_one_step(
                                self.client_timer.local_outer_epoch_idx,
                                self.client_timer.local_inner_iter_idx,
                                self.check_end_epoch(),
                                self.train_tracker,
                                self.metrics
                            )
                    local_sample_number = self.worker.local_sample_number
                    # batch_metric_stat = self.metrics.evaluate(loss, output, target)
                    # self.train_tracker.update_metrics(batch_metric_stat, n_samples=target.size(0))
                    # logging.debug('(Local Training Epoch: {}, Iter: {} '.format(
                    #     epoch, iteration) + self.metrics.str_fn(batch_metric_stat))
                    # begin to send model
                    logging.debug("Begin send and receive")
                    # if not self.sync_receive_all_event.is_set():
                    #     self.sync_receive_all_event.clear()
                    client_other_params = {}
                    for neighbor_idx in self.topology_manager.get_out_neighbor_idx_list(self.client_index):
                        self.send_result_to_neighbors(neighbor_idx, self.model_trainer.get_model_params(),
                                                    local_sample_number, client_other_params)
                    # wait for receiving all
                    self.sync_receive_all_event.wait()

                    self.neighbor_transfer_lock.acquire()

                    aggregated_model_params = self.worker.aggregate()
                    # local model is updated in aggregate()
                    self.worker.set_model_params(aggregated_model_params)
                    if self.neighbor_transfer_lock.locked():
                        self.neighbor_transfer_lock.release()
                    # has completed aggregation, allow receive others' models 
                    # self.complete_aggregation_event.set()

                    # logging.info('Local Training Epoch: {} iter: {} \t Loss: {:.6f}, Acc1: {:.6f}'.format(
                    #                 epoch, iteration, batch_metric_stat['Loss'], batch_metric_stat['Acc1']))

                    self.start_epoch_event.clear()
                    self.sync_receive_all_event.clear()

                    client_other_params = {}
                    self.test_and_send_to_coordinator(client_other_params)
                    self.client_timer.past_iterations(iterations=1)
                # self.model_trainer.lr_schedule(epoch+1)

    def run_consensus(self):
        with raise_MPI_error(MPI):
            self.consensus_tensor = torch.ones([10]) * self.rank
            for iteration in range(500):
                logging.debug("wait start_epoch_event")
                self.start_epoch_event.wait()

                # begin to send model
                logging.debug("Begin send and receive")
                client_other_params = {}
                for neighbor_idx in self.topology_manager.get_out_neighbor_idx_list(self.client_index):
                    self.send_result_to_neighbors(neighbor_idx, self.consensus_tensor, 0, client_other_params)
                # wait for receiving all
                self.sync_receive_all_event.wait()

                self.neighbor_transfer_lock.acquire()
                aggregated_model_params = self.worker.aggregate_tensor(self.consensus_tensor)
                # local model is updated in aggregate()
                self.consensus_tensor = aggregated_model_params
                if self.neighbor_transfer_lock.locked():
                    self.neighbor_transfer_lock.release()
                # has completed aggregation, allow receive others' models 

                if self.rank == 0:
                    with get_lock(self.total_metric_lock):
                        self.flag_client_finish_dict[self.rank] = True
                        self.check_worker_finish_and_notify()
                else:
                    self.send_notify_to_coordinator(0)
                logging.info('Local iter: {} \t Tensor: {}'.format(
                                iteration, self.consensus_tensor))

                self.start_epoch_event.clear()
                self.sync_receive_all_event.clear()


    # TODO not finished
    def run_async(self):
        local_sample_number, _ = self.worker.train_one_step()
        # =========================== enter lock
        self.neighbor_transfer_lock.acquire()
        if self.worker.check_whether_any_receive():
            aggregated_model_params = self.worker.aggregate()
            # local model is updated in aggregate()
            self.worker.set_model_params(aggregated_model_params)
        if self.neighbor_transfer_lock.locked():
            self.neighbor_transfer_lock.release()
        # =========================== exit lock

        # this event is actually not used in async
        self.complete_aggregation_event.set()
        client_other_params = {}
        for neighbor_idx in self.topology_manager.get_out_neighbor_idx_list(self.client_index):
            self.send_result_to_neighbors(neighbor_idx, aggregated_model_params, local_sample_number, client_other_params)


