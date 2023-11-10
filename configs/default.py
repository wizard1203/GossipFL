import os
import random

# from .config import CfgNode as CN
from .config import CfgNode as CN

_C = CN()



# ---------------------------------------------------------------------------- #
# version control
# This is used to filter experiment results.
# ---------------------------------------------------------------------------- #
_C.version = 'v1.0'
# _C.bug_fix = 'fix_local_round_update'
_C.fix_local_round_update = True


# ---------------------------------------------------------------------------- #
# wandb settings
# ---------------------------------------------------------------------------- #
_C.entity = None
_C.project = 'ddl-bench'

# ---------------------------------------------------------------------------- #
# mode settings
# ---------------------------------------------------------------------------- #
_C.mode = 'distributed'  # distributed or standalone or centralized


# ---------------------------------------------------------------------------- #
# distributed settings
# ---------------------------------------------------------------------------- #
_C.client_num_in_total = 100
_C.client_num_per_round = 8


# ---------------------------------------------------------------------------- #
# device settings
# ---------------------------------------------------------------------------- #
_C.is_mobile = 0


# ---------------------------------------------------------------------------- #
# cluster settings
# ---------------------------------------------------------------------------- #
_C.rank = 0
_C.client_index = 0
_C.gpu_server_num = 1
_C.gpu_util_file = None
_C.gpu_util_key = None
_C.gpu_util_parse = None
_C.cluster_name = None

_C.gpu_index = 0  # for centralized training or standalone usage

# ---------------------------------------------------------------------------- #
# task settings
# ---------------------------------------------------------------------------- #
_C.task = 'classification' #    ["classification", "stackoverflow_lr", "ptb"]




# ---------------------------------------------------------------------------- #
# dataset
# ---------------------------------------------------------------------------- #
_C.dataset = 'cifar10'
_C.data_dir = './../../../data/cifar10'
_C.partition_method = 'iid'
_C.partition_alpha = 0.5
_C.if_timm_dataset = False
_C.data_load_num_workers = 4

_C.an4_audio_path = " " # an4 audio paht
_C.lstm_num_steps = 35 # used for ptb, lstm_num_steps
_C.lstm_clip_grad = True
_C.lstm_clip_grad_thres = 0.25
_C.lstm_embedding_dim = 8
_C.lstm_hidden_size = 256

# ---------------------------------------------------------------------------- #
# data sampler
# ---------------------------------------------------------------------------- #
_C.data_sampler = "Random"   #  ["imbalance", "decay_imb"]

# _C.imbalance_sampler = False # we discard this, using data_sampler to indicate eveything.
_C.imbalance_beta = 0.9999
_C.imbalance_beta_min = 0.8
_C.imbalance_beta_decay_rate = 0.992
# ["global_round", "local_round", "epoch"]
_C.imbalance_beta_decay_type = "global_round"

# ---------------------------------------------------------------------------- #
# data_preprocessing
# ---------------------------------------------------------------------------- #
_C.data_transform = "NormalTransform"  # or FLTransform


# ---------------------------------------------------------------------------- #
# checkpoint_save
# ---------------------------------------------------------------------------- #
_C.checkpoint_save = False
_C.checkpoint_save_model = False
_C.checkpoint_save_optim = False
_C.checkpoint_save_train_metric = False
_C.checkpoint_save_test_metric = False
_C.checkpoint_root_path = "./checkpoints/"
_C.checkpoint_epoch_list = [10, 20, 30]
_C.checkpoint_file_name_save_list = ["model", "dataset"]

# ---------------------------------------------------------------------------- #
# correlation compare layer list
# ---------------------------------------------------------------------------- #
_C.corr_layers_list = ["layer1", "layer2"]
_C.corr_dataset_len = 100



# ---------------------------------------------------------------------------- #
# model
# ---------------------------------------------------------------------------- #
_C.model = 'resnet20'
_C.pretrained = False
_C.pretrained_dir = " "

# refer to https://github.com/kevinhsieh/non_iid_dml/blob/master/apps/caffe/examples/cifar10/5parts/resnetgn20_train_val.prototxt.template
_C.group_norm_num = 2


# ---------------------------------------------------------------------------- #
# Imbalance weight
# ---------------------------------------------------------------------------- #
_C.imbalance_loss_reweight = False

# ---------------------------------------------------------------------------- #
# loss function
# ---------------------------------------------------------------------------- #
_C.loss_fn = 'CrossEntropy'
""" ['CrossEntropy', 'nll_loss', 'LDAMLoss', 'local_LDAMLoss',
        'FocalLoss', 'local_FocalLoss']
"""

# _C.LDAMLoss_independent = True

# ---------------------------------------------------------------------------- #
# trainer
#---------------------------------------------------------------------------- #
# ['normal',  'lstm', 'nas']
_C.trainer_type = 'normal'


# ---------------------------------------------------------------------------- #
# algorithm settings
# ---------------------------------------------------------------------------- #
_C.algorithm = 'PSGD'
_C.psgd_exchange = 'grad' # 'grad', 'model'
_C.psgd_grad_sum = False
_C.psgd_grad_debug = False
_C.if_get_diff = False # this is suitable for other PS algorithms
_C.exchange_model = True

# Asynchronous PSGD
# _C.apsgd_exchange = 'grad' # 'grad', 'model' # discarded, use psgd_exchange

# Local SGD
_C.local_round_num = 4

# CHOCO SGD
_C.consensus_stepsize = 0.5

# SAPS FL
_C.bandwidth_type = 'random' # 'random' 'real'
_C.B_thres = 3.0
_C.T_thres = 3



# torch_ddp
_C.local_rank = 0
_C.init_method = 'tcp://127.0.0.1:23456'


# hvd settings and maybe used in future
_C.FP16 = False
_C.logging_gradients = False
_C.merge_threshold = 0
# horovod version
_C.hvd_origin = False
_C.nsteps_update = 1
_C.hvd_momentum_correction = 0 # Set it to 1 to turn on momentum_correction
_C.hvd_is_sparse = False






# fedprox
_C.fedprox_mu = 0.1



# ---------------------------------------------------------------------------- #
# compression Including:
# 'topk','randomk', 'gtopk', 'randomkec',  'eftopk', 'gtopkef'
# 'quantize', 'qsgd', 'sign'
# ---------------------------------------------------------------------------- #
_C.compression = None

_C.compress_ratio = 1.0
_C.quantize_level = 32
_C.is_biased = 0

# ---------------------------------------------------------------------------- #
# optimizer settings
# comm_round is only used in FedAvg.
# ---------------------------------------------------------------------------- #
_C.epochs = 90
_C.comm_round = 10
_C.client_optimizer = 'no' # Please indicate which optimizer is used, if no, set it as 'no'
_C.server_optimizer = 'no'
_C.batch_size = 32
_C.lr = 0.1
_C.wd = 0.0001
_C.momentum = 0.0
_C.nesterov = False
_C.clip_grad = False



# ---------------------------------------------------------------------------- #
# Learning rate schedule parameters
# ---------------------------------------------------------------------------- #
_C.sched = 'no'   # no (no scheudler), StepLR MultiStepLR  CosineAnnealingLR
_C.lr_decay_rate = 0.992
_C.step_size = 1
_C.lr_milestones = [30, 60]
_C.lr_T_max = 10
_C.lr_eta_min = 0
_C.warmup_epochs = 0


# ---------------------------------------------------------------------------- #
# Regularation
# ---------------------------------------------------------------------------- #



# ---------------------------------------------------------------------------- #
# Evaluate settings
# ---------------------------------------------------------------------------- #
_C.frequency_of_the_test = 1



# ---------------------------------------------------------------------------- #
# Robust test
# ---------------------------------------------------------------------------- #
_C.Failure_chance = None



# ---------------------------------------------------------------------------- #
# logging
# ---------------------------------------------------------------------------- #
_C.level = 'INFO' # 'INFO' or 'DEBUG'




# ---------------------------------------------------------------------------- #
# other settings
# ---------------------------------------------------------------------------- #
_C.ci = 0
_C.seed = 0






