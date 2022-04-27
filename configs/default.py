from logging import FATAL
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
_C.exp_mode = 'debug'    # debug or normal or ready


# ---------------------------------------------------------------------------- #
# wandb settings
# ---------------------------------------------------------------------------- #
_C.entity = None
_C.project = 'test'
_C.wandb_upload_client_list = [0, 1] # 0 is the server
_C.wandb_save_record_dataframe = False
_C.wandb_offline = False
_C.wandb_record = True


# ---------------------------------------------------------------------------- #
# mode settings
# ---------------------------------------------------------------------------- #
_C.mode = 'distributed'  # distributed


# ---------------------------------------------------------------------------- #
# distributed settings
# ---------------------------------------------------------------------------- #
_C.client_num_in_total = 100
_C.client_num_per_round = 10
_C.instantiate_all = True
_C.clear_buffer = True

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
_C.dataset_aug = "default"
_C.dataset_resize = False
_C.dataset_load_image_size = 32
_C.num_classes = 10
_C.data_efficient_load = True    #  Efficiently load dataset, only load one full dataset, but split to many small ones.
_C.data_save_memory_mode = False    #  Clear data iterator, for saving memory, but may cause extra time.
_C.data_dir = './../../../data/cifar10'
_C.partition_method = 'iid'
_C.partition_alpha = 0.5
_C.dirichlet_min_p = None #  0.001    set dirichlet min value for letting each client has samples of each label
_C.dirichlet_balance = False # This will try to balance dataset partition among all clients to make them have similar data amount

_C.load_multiple_generative_dataset_list = ["style_GAN_init"]



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
_C.imbalance_sample_warmup_rounds = 0

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


_C.image_save_path = "./checkpoints/"

# ---------------------------------------------------------------------------- #
# record config
# ---------------------------------------------------------------------------- #
_C.record_dataframe = False
_C.record_level = 'epoch'   # iteration



# ---------------------------------------------------------------------------- #
# Losses track
# ---------------------------------------------------------------------------- #
_C.losses_track = False
_C.losses_track_client_list = [0, 1]
_C.losses_curve_2model = False
_C.losses_curve_2model_selected_client = [0, 1]
_C.losses_curve_2model_comm_round_list = [0, 30, 50, 100, 150, 199]




# ---------------------------------------------------------------------------- #
# Param Importance Measure
# ---------------------------------------------------------------------------- #
_C.param_track = False
_C.param_nonzero_ratio = 1.0
_C.param_track_with_training = False
_C.param_track_max_iters = 'max'     # means whole dataset

# Use these three args alternatively to decide which layers should be tracked.
_C.param_track_layers_list = [""]
_C.param_track_layers_length = -1
_C.param_track_types = ["Conv2d","Linear"] #  [""] or   ["Conv2d","Linear"]

_C.param_track_wandb_print_layers = -1    # determine how many layers to output, -1 means all
_C.param_track_save_pth_epoch_list = [0,1,2,3,4,9,14,19,24,29,39,59,79,99]
_C.param_track_batch_size = 64

# _C.param_crt = "weight"

# "weight","weight_abs",""
# "E_GiWi","Taylor_SO",'V_GiWi_aprx','Taylor_SObyFO'
# "grad_mean","grad_std","weight_mean","weight_std"
_C.param_crt_list = ["weight","V_GiWi_aprx"]





# ---------------------------------------------------------------------------- #
# model
# ---------------------------------------------------------------------------- #
_C.model = 'resnet20'
_C.model_input_channels = 3
_C.model_out_feature = False
_C.model_out_feature_layer = "last"
_C.model_feature_dim = 512
_C.model_output_dim = 10
_C.pretrained = False
_C.pretrained_dir = " "

# refer to https://github.com/kevinhsieh/non_iid_dml/blob/master/apps/caffe/examples/cifar10/5parts/resnetgn20_train_val.prototxt.template
_C.group_norm_num = 0


# ---------------------------------------------------------------------------- #
# generator
# ---------------------------------------------------------------------------- #
_C.image_resolution = 32
_C.style_gan_ckpt = ""
_C.style_gan_style_dim = 64   #  512
_C.style_gan_n_mlp = 1
_C.style_gan_cmul = 1
_C.style_gan_sample_z_mean = 0.3
_C.style_gan_sample_z_std = 0.1

_C.vae_decoder_z_dim = 8
_C.vae_decoder_ngf = 64



# ---------------------------------------------------------------------------- #
# generative_dataset
# ---------------------------------------------------------------------------- #

_C.generative_dataset_load_in_memory = False
_C.generative_dataset_pin_memory = True
_C.generative_dataset_shared_loader = False          # For efficiently loading, but may cause bugs.

_C.generative_dataset_root_path = './../../../data/generative'
_C.generative_dataset_resize = None            #     resize image
_C.generative_dataset_grayscale = False           # Gray Scale



# ---------------------------------------------------------------------------- #
# Average weight
# ---------------------------------------------------------------------------- #
"""[even, datanum, inv_datanum, inv_datanum2datanum, even2datanum,
        ]
"""
# datanum2others is not considerred for now.
_C.fedavg_avg_weight_type = 'datanum'   

# Friendly Averaging
_C.friend_avg = False
_C.friend_avg_crt = "E_grad_pow2_aprx"   #   Criteria: ['V_grad_aprx','E_grad_pow2_aprx','']
_C.friend_avg_layers = []           # Indicate apply friend avg on which layers, or all. 
_C.friend_avg_min_crt = 1e-6
_C.friend_avg_normalize = "minmax"   #  max, min_limit, no, 
_C.friend_avg_temperature = 0.001
_C.friend_avg_rand_pertub = 0.01     #  friend_avg_rand_pertub






# ---------------------------------------------------------------------------- #
# Dif local steps
# ---------------------------------------------------------------------------- #
_C.fedavg_local_step_type = 'whole'   # whole, fixed, fixed2whole
_C.fedavg_local_step_fixed_type = 'lowest'   # default, lowest, highest, averaged
_C.fedavg_local_step_num = 10    # used for the fixed local step default 




# ---------------------------------------------------------------------------- #
# loss function
# ---------------------------------------------------------------------------- #
_C.loss_fn = 'CrossEntropy'
""" ['CrossEntropy', 'nll_loss', 'LDAMLoss', 'local_LDAMLoss',
        'FocalLoss', 'local_FocalLoss']
"""
_C.normal_supcon_loss = False



# ---------------------------------------------------------------------------- #
# Imbalance weight
# ---------------------------------------------------------------------------- #
_C.imbalance_loss_reweight = False



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
_C.fedprox = False
_C.fedprox_mu = 0.1

# fedavg
_C.fedavg_label_smooth = 0.0


# scaffold
_C.scaffold = False



# ---------------------------------------------------------------------------- #
# x_noniid_measure
# ---------------------------------------------------------------------------- #
_C.x_noniid_measure = "no"
_C.x_noniid_measure_dlmodel = "vgg9"


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
_C.max_epochs = 90
_C.global_epochs_per_round = 1
_C.comm_round = 90
_C.client_optimizer = 'no' # Please indicate which optimizer is used, if no, set it as 'no'
_C.server_optimizer = 'no'
_C.batch_size = 32
_C.lr = 0.1
_C.wd = 0.0001
_C.momentum = 0.9
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
_C.lr_warmup_type = 'constant' # constant, gradual.
_C.warmup_epochs = 0
_C.lr_warmup_value = 0.1



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






