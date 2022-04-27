import argparse
import logging
import os
import random
import socket
import sys
import yaml
import pickle
import traceback

import matplotlib.pyplot as plt
import numpy as np
import psutil
import setproctitle
import torch
import wandb

from torchscan  import summary

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))



from configs import get_cfg, build_config
from utils.logger import (
    logging_config
)
from model.build import create_model




def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    parser.add_argument("--config_file", default=None, type=str)
    parser.add_argument("--config_name", default=None, type=str,
                        help="specify add which type of config")
    parser.add_argument("--extra_name", default=None, type=str,
                        help="specify extra name of checkpoint")
    parser.add_argument("--execute", default="cca_compare", type=str,
                        help="specify operation")

    parser.add_argument("opts", help="Modify config options using the command-line",
                        default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args

if __name__ == "__main__":


    process_id = 0
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

    cfg.rank = 0

    # show ultimate config
    logging.info(dict(cfg))

    # customize the process name
    str_process_name = "model scan: " + str(process_id)
    setproctitle.setproctitle(str_process_name)

    logging_config(args=cfg, process_id=process_id)

    hostname = socket.gethostname()
    logging.info("#############process ID = " + str(process_id) +
                ", host name = " + hostname + "########" +
                ", process ID = " + str(os.getpid()) +
                ", process Name = " + str(psutil.Process(os.getpid())))



    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic =True

    device = torch.device("cuda:" + str(cfg.gpu_index) if torch.cuda.is_available() else "cpu")

    # create model.
    if cfg.dataset in ["cifar10", "cifar100"]:
        class_num = 10
        sample = (3, 32, 32)
        # sample = (28, 28, 3)
    elif cfg.dataset in ["mnist", "femnist"]:
        class_num = 10
        sample = (1, 28, 28)
        cfg.model_input_channels = 1
    elif cfg.dataset in ["imagenet"]:
        class_num = 1000
        sample = (3, 224, 224)
    else:
        raise NotImplementedError 
    model = create_model(args=cfg, model_name=cfg.model, output_dim=class_num)
    summary(model, sample)
    for name, params in model.named_parameters():
        print("param name ---- {}: shape ---- {}".format(name, params.shape))
