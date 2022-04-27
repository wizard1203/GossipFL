import argparse
import logging
import os
import random
import socket
import sys
import yaml

import traceback
from mpi4py import MPI

import numpy as np
import psutil
import setproctitle
import torch
import wandb
# add the FedML root directory to the python path

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))


from data_preprocessing.build import load_data
from model.build import create_model
from trainers.build import create_trainer

from utils.context import (
    raise_MPI_error
)
from utils.logger import (
    logging_config
)
from utils.gpu_mapping import (
    init_training_device_from_gpu_util_file,
    init_training_device_from_gpu_util_parse
)

from configs import get_cfg, build_config
from algorithms.fedavg.FedAvgAPI import FedML_init, FedML_FedAvg_distributed


from utils.data_utils import (
    get_local_num_iterations,
    get_avg_num_iterations
)


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    parser.add_argument("--config_file", default=None, type=str)
    parser.add_argument("--config_name", default=None, type=str,
                        help="specify add which type of config")
    parser.add_argument("opts", help="Modify config options using the command-line",
                        default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # initialize distributed computing (MPI)
    comm, process_id, worker_number = FedML_init()

    with raise_MPI_error(MPI):
        # parse python script input parameters
        parser = argparse.ArgumentParser()
        args = add_args(parser)

        #### set up cfg ####
        # default cfg
        cfg = get_cfg()

        # add registered cfg
        # some arguments that are needed by build_config come from args.
        cfg.setup(args)
        build_config(cfg, args.config_name)

        # Build config once again
        cfg.setup(args)

        cfg.rank = process_id
        if cfg.algorithm in ['FedAvg', 'PSGD']:
            if process_id == 0:
                cfg.role = 'server'
                cfg.server_index = process_id
                cfg.client_index = -1
            else:
                cfg.role = 'client'
                cfg.server_index = -1
                cfg.client_index = process_id - 1

        elif cfg.algorithm in ['DPSGD', 'DCD_PSGD', 'CHOCO_SGD', 'SAPS_FL']:
            cfg.role = 'client'
            cfg.server_index = -1
            cfg.client_index = process_id
        else:
            raise NotImplementedError


        seed = cfg.seed

        # show ultimate config
        logging.info(dict(cfg))

        # customize the process name
        str_process_name = cfg.algorithm + " (distributed):" + str(process_id)
        setproctitle.setproctitle(str_process_name)

        logging_config(args=cfg, process_id=process_id)

        hostname = socket.gethostname()
        logging.info("#############process ID = " + str(process_id) +
                    ", host name = " + hostname + "########" +
                    ", process ID = " + str(os.getpid()) +
                    ", process Name = " + str(psutil.Process(os.getpid())))

        # initialize the wandb machine learning experimental tracking platform (https://www.wandb.com/).
        if process_id == 0:
            if cfg.wandb_record:
                wandb.init(
                    # settings=wandb.Settings(start_method="fork"),
                    entity=cfg.entity,
                    project=cfg.project,
                    name=cfg.algorithm + " (d)" + str(cfg.partition_method) + "-" +str(cfg.dataset)+
                        "-r" + str(cfg.comm_round) +
                        "-e" + str(cfg.max_epochs) + "-" + str(cfg.model) + "-" +
                        str(cfg.client_optimizer) + "-bs" + str(cfg.batch_size) +
                        "-lr" + str(cfg.lr) + "-wd" + str(cfg.wd),
                    config=dict(cfg)
                )
        if cfg.wandb_offline:
            os.environ['WANDB_MODE'] = 'dryrun'

        # Set the random seed. The np.random seed determines the dataset partition.
        # The torch_manual_seed determines the initial weight.
        # We fix these two, so that we can reproduce the result.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic =True

        logging.info("process_id = %d, size = %d" % (process_id, worker_number))
        if cfg.gpu_util_parse is not None:
            device, gpu_util_map = init_training_device_from_gpu_util_parse(process_id, worker_number, cfg.gpu_util_parse)
        else:
            device, gpu_util_map = init_training_device_from_gpu_util_file(process_id, worker_number, cfg.gpu_util_file, cfg.gpu_util_key)

        # load data
        # dataset = load_data(cfg, cfg.dataset)
        if cfg.partition_method in ["iid", "homo"]:
            task = "distributed"
        else:
            task = "federated"

        if cfg.algorithm in ['FedAvg', 'AFedAvg', 'PSGD', 'APSGD', 'Local_PSGD', 'FedSGD']:
            mode = "PS-distributed"
        elif cfg.algorithm in ['DPSGD', 'DCD_PSGD', 'CHOCO_SGD', 'SAPS_FL']:
            mode = "Gossip-distributed"
        else:
            raise NotImplementedError

        dataset = load_data(
            load_as="training", args=cfg, process_id=process_id, mode=mode, task=task, data_efficient_load=True,
            dirichlet_balance=cfg.dirichlet_balance, dirichlet_min_p=cfg.dirichlet_min_p,
            dataset=cfg.dataset, datadir=cfg.data_dir,
            partition_method=cfg.partition_method, partition_alpha=cfg.partition_alpha,
            client_number=cfg.client_num_in_total, batch_size=cfg.batch_size, num_workers=cfg.data_load_num_workers,
            data_sampler=cfg.data_sampler,
            resize=cfg.dataset_load_image_size, augmentation=cfg.dataset_aug)

        [train_data_num, test_data_num, train_data_global, test_data_global,
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, other_params] = dataset

        # create model.
        # Note if the model is DNN (e.g., ResNet), the training will be very slow.
        # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
        # model = create_model(cfg, model_name=cfg.model, output_dim=dataset[7], **other_params)
        model = create_model(cfg, model_name=cfg.model, output_dim=cfg.model_output_dim,
                            pretrained=cfg.pretrained, **other_params)

        num_iterations = get_avg_num_iterations(train_data_local_num_dict, cfg.batch_size)
        # model_trainer = create_trainer(
        #     cfg, device, model, num_iterations=num_iterations, train_data_num=train_data_num,
        #     test_data_num=test_data_num, train_data_global=train_data_global, test_data_global=test_data_global,
        #     train_data_local_num_dict=train_data_local_num_dict, train_data_local_dict=train_data_local_dict,
        #     test_data_local_dict=test_data_local_dict, class_num=class_num, other_params=other_params)
        init_state_kargs = {}
        model_trainer = create_trainer(
            cfg, device, model, num_iterations=num_iterations,
            server_index=cfg.server_index, client_index=cfg.client_index, role=cfg.role,
            **init_state_kargs)

        if cfg.algorithm == 'FedAvg':
            FedML_FedAvg_distributed(process_id, worker_number, device, comm,
                                    model, train_data_num, train_data_global, test_data_global,
                                    train_data_local_num_dict, train_data_local_dict, test_data_local_dict, cfg,
                                    model_trainer)
        elif cfg.algorithm == 'PSGD':
            from algorithms.PSGD.PSGD_API import FedML_init, FedML_PSGD_distributed
            FedML_PSGD_distributed(process_id, worker_number, device, comm,
                                    model, train_data_num, train_data_global, test_data_global,
                                    train_data_local_num_dict, train_data_local_dict, test_data_local_dict, cfg,
                                    model_trainer)
        elif cfg.algorithm == 'DPSGD':
            from algorithms.DPSGD.DPSGD_API import FedML_init, FedML_DPSGD
            FedML_DPSGD(process_id, worker_number, device, comm,
                                model, train_data_num, train_data_global, test_data_global,
                                train_data_local_num_dict, train_data_local_dict, test_data_local_dict, cfg,
                                model_trainer)
        elif cfg.algorithm == 'DCD_PSGD':
            from algorithms.DCD_PSGD.DCD_PSGD_API import FedML_init, FedML_DCD_PSGD
            FedML_DCD_PSGD(process_id, worker_number, device, comm,
                                model, train_data_num, train_data_global, test_data_global,
                                train_data_local_num_dict, train_data_local_dict, test_data_local_dict, cfg,
                                model_trainer)
        elif cfg.algorithm == 'CHOCO_SGD':
            from algorithms.CHOCO_SGD.CHOCO_SGD_API import FedML_init, FedML_CHOCO_SGD
            FedML_CHOCO_SGD(process_id, worker_number, device, comm,
                                model, train_data_num, train_data_global, test_data_global,
                                train_data_local_num_dict, train_data_local_dict, test_data_local_dict, cfg,
                                model_trainer)
        elif cfg.algorithm == 'SAPS_FL':
            from algorithms.SAPS_FL.SAPS_FL_API import FedML_init, FedML_SAPS_FL
            FedML_SAPS_FL(process_id, worker_number, device, comm,
                                model, train_data_num, train_data_global, test_data_global,
                                train_data_local_num_dict, train_data_local_dict, test_data_local_dict, cfg,
                                model_trainer)
        else:
            raise NotImplementedError







