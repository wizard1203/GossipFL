from __future__ import print_function
import os
import sys
import logging
import time
import copy
import datetime
import itertools
import numpy as np

from pandas import Series,DataFrame
import pandas as pd

import wandb


PSGD = 'PSGD'
APSGD = 'APSGD'
LOCAL_PSGD = 'Local_PSGD'
FEDAVG = 'FedAvg'
FEDNOVA = 'FedNova'
DPSGD = "DPSGD"
DCD = 'DCD_PSGD'
CHOCO_SGD = 'CHOCO_SGD'
SAPS = 'SAPS_FL'

SCAFFOLD = 'scaffold'
FEDPROX = 'fedprox'

CENTRALIZED = 'centralized'

algorithm_list = [
    PSGD,
    APSGD,
    LOCAL_PSGD ,
    FEDAVG,
    FEDNOVA,
    DPSGD,
    DCD,
    CHOCO_SGD,
    SAPS,
    CENTRALIZED,
]

eftopk = 'eftopk'
topk = 'topk'
randomk = 'randomk'
qsgd = "qsgd"


compression_list = [
    eftopk,
    topk,
    randomk ,
    qsgd,
]


sgd = "sgd"
fedprox = "FedProx"
optimizer_list = [sgd, fedprox]


resnet10_v2 = 'resnet10_v2'
resnet18_v2 = 'resnet18_v2'
resnet34_v2 = 'resnet34_v2'
resnet50_v2 = 'resnet50_v2'

resnet18 = 'resnet18'
resnet20 = 'resnet20'
resnet50 = 'resnet50'



vgg9 = 'vgg-9'
mnistnet = 'mnistflnet'
cifar10flnet = 'cifar10flnet'
SVCCAConvNet = 'SVCCAConvNet'
model_list = [resnet20, resnet10_v2, resnet18_v2, resnet34_v2, resnet50_v2,
            vgg9, mnistnet,
            cifar10flnet, SVCCAConvNet,
            resnet50]


cca_resnet20_layers = [\
    'stage_1.0.conv_b','stage_1.1.conv_b','stage_1.2.conv_b',\
    'stage_2.0.conv_b','stage_2.1.conv_b','stage_2.2.conv_b',\
    'stage_3.0.conv_b','stage_3.1.conv_b','stage_3.2.conv_b',\
    'classifier'\
]

# resnet20_layers = [\
#     'stage_1.0.conv_b.weight','stage_1.1.conv_b.weight','stage_1.2.conv_b.weight',\
#     'stage_2.0.conv_b.weight','stage_2.1.conv_b.weight','stage_2.2.conv_b.weight',\
#     'stage_3.0.conv_b.weight','stage_3.1.conv_b.weight','stage_3.2.conv_b.weight',\
#     'classifier.weight'\
# ]


cca_cifar10flnet_layers = [\
    'conv1','pool1','norm1',\
    'conv2','pool2','norm2',\
    'fc1','fc2','fc3'\
]

cca_SVCCAConvNet_layers = [\
'conv1',\
'conv2',\
'bn1',\
'pool1',\
'conv3',\
'conv4',\
'conv5',\
'bn2',\
'pool2',\
'fc1',\
'bn3',\
'fc2',\
'bn4',\
]

cca_model_layers = {
    resnet20: cca_resnet20_layers,
    cifar10flnet: cca_cifar10flnet_layers,
    SVCCAConvNet: cca_SVCCAConvNet_layers,
}

track_resnet20_layers = [\
    'stage_1.0.conv_b','stage_1.1.conv_b','stage_1.2.conv_b',\
    'stage_2.0.conv_b','stage_2.1.conv_b','stage_2.2.conv_b',\
    'stage_3.0.conv_b','stage_3.1.conv_b','stage_3.2.conv_b',\
    'classifier'\
]

track_cifar10flnet_layers = [\
    'conv1',\
    'conv2',\
    'fc1','fc2','fc3'\
]

track_SVCCAConvNet_layers = [\
'conv1',\
'conv2',\
'conv3',\
'conv4',\
'conv5',\
'fc1',\
'fc2',\
]






track_model_layers = {
    resnet20: track_resnet20_layers,
    cifar10flnet: track_cifar10flnet_layers,
    SVCCAConvNet: track_SVCCAConvNet_layers
}





mnist = "mnist"
fmnist = 'fmnist'
cifar10 = 'cifar10'
cifar100 = 'cifar100'
tiny_imagenet_200 = 'Tiny-ImageNet-200'
SVHN = "SVHN"
dataset_list = [mnist, cifar10, fmnist, tiny_imagenet_200, SVHN]

# cifar10flnet_name = 'cifar10flnet'
# resnet20_name = 'resnet20'


# random_sampler_name = "Random"
data_sampler_list = ["Random", "decay_imb", "imbalance"]
# imbalnace_beta_min_list = [0.99, 0.98, 0.95, 0.8]
# imbalnace_beta_min_list = [0.99, 0.98, 0.95, 0.8]
imbalnace_beta_min_list = [0.99, 0.98, 0.95, 0.8]

loss_list = ["CrossEntropy", "LDAMLoss"]
# loss_list = ["CrossEntropy", "LDAMLoss"]


partition_alpha_list = [0.99, 0.9, 0.8]



client_num_in_total_list = [10] # '32'


mnist_size = 50000
cifar10_size = 50000

TRAIN = "Train"
TEST = "Test"

ACC = "Acc1"
LOSS = "Loss"

TRAIN_ACC = "Train/Acc1"
TEST_ACC = "Test/Acc1"

LOSSES = "losses"
GRAD_NORM = "grad_norm"
GRAD_SUM_NORM = "grad_sum_norm"
MODEL_DIFF_NORM = "model_diff_norm"
MODEL_DIF_WHOLE_NORM = "model_dif_whole_norm"
MODEL_ROTATION_WHOLE_NORM = "model_rotation_whole_norm"
MODEL_LAYER_SVD_SIMILARITY = "model_layer_SVD_similarity"
MODEL_LAYER_COSINE_SIMILARITY = "model_layer_Cosine_similarity"

LP = "LP"

SVCCA = "svcca"

things_on_figure = {
    ACC: "Top-1 Test Accuracy [%]",
    LOSSES: "losses",
    GRAD_NORM: "Accumulated Norms of Grads",
    GRAD_SUM_NORM: "Norm of Accumulated Grads",
    MODEL_DIFF_NORM: "Norm of Model Diff",
    MODEL_DIF_WHOLE_NORM: "Norm of Model Whole Diff",
    MODEL_ROTATION_WHOLE_NORM: "Norm of Model Rotation",
    MODEL_LAYER_SVD_SIMILARITY: "SVD similarity of layer weights",
    MODEL_LAYER_COSINE_SIMILARITY: "Cosine similarity of layer weights",
    SVCCA: "SVCCA similarity",
}


max_epochs_dict = {
    mnistnet: 200,
    cifar10flnet: 400,
    resnet20: 300,
    SVCCAConvNet: 400,
}

model_name_on_figure = {
    mnistnet: 'mnistCNN',
    cifar10flnet: 'cifar10CNN',
    resnet20: 'resnet20',
    SVCCAConvNet: 'SVCCAConvNet',
}



iteration_name = "iteration"
round_name = "global_comm_round"
epoch_name = "epoch"

iid_target_acc = {
    mnistnet: 98.0,
    cifar10flnet: 60.0,
    resnet20: 80.0
}
noniid_target_acc = {
    mnistnet: 95.0,
    cifar10flnet: 30.0,
    resnet20: 80.0
}

noniid_target_acc_14_worker = {
    mnistnet: 98.0,
    cifar10flnet: 30.0,
    resnet20: 80.0
}



def build_get_data_func(alias_run_map, x_name, max_x_dict, max_x_key):

    def get_data_func(exp_run, alias, config, help_params, line_params):
        try:
            history = exp_run.history
            # logging.info(history)
            # logging.info("In get_epoch_func, alias: {}, history: {}".format(
            #     alias, history))

            y_name = line_params["metric_thing"]

            x = np.array(list(history.loc[(history[x_name]<max_x_dict[config[max_x_key]]) &\
                (history[y_name].notnull())][x_name])
            )

            # max_x = max(x)
            max_x = np.max(x)
            alias_run_map[alias]["max_"+x_name] = max_x
            y = np.array(list(history.loc[(history[x_name]<max_x_dict[config[max_x_key]]) &\
                (history[y_name].notnull())][y_name])
            )

            # max_y = max(y)
            max_y = np.max(y)
            max_y_index = np.argmax(y)
            alias_run_map[alias][y_name] = y[-1]
            alias_run_map[alias]["final_"+y_name] = y[-1]
            alias_run_map[alias]["max_"+y_name] = max_y
            alias_run_map[alias]["max_index"] = max_y_index
            alias_run_map[alias]["run_uid"] = exp_run.uid

            if "target_acc" in line_params:
                target_acc = line_params["target_acc"]

                high_acc_epochs = np.argwhere(y > target_acc)
                if len(high_acc_epochs) > 0:
                    first_epoch = x[high_acc_epochs[0]][0]
                    # first_epoch = x[high_acc_epochs[0]]
                else:
                    first_epoch = 0
                alias_run_map[alias]["target_acc_epoch"] = first_epoch
                alias_run_map[alias]["target_acc"] = target_acc
                # logging.info("In get_data_func, alias: {}, x: {}, max_x: {}, y: {}, max_y:{}".format(
                #     alias, x, max_x, y, max_y))
            return x, max_x, y, max_y, None, None
        except:
            logging.info(f"\n  Ploting, not find history {alias} \n")
            logging.info(f"config:{config}, exp_run.uid")
            x = np.array(list(range(10)))
            y = np.array(list(range(10)))
            max_x = 0.0
            max_y = 0.0
            y_name = line_params["metric_thing"]
            alias_run_map[alias]["max_"+x_name] = 0
            alias_run_map[alias][y_name] = 0
            alias_run_map[alias]["final_"+y_name] = 0.0
            alias_run_map[alias]["max_"+y_name] = 0
            alias_run_map[alias]["max_index"] = 0
            alias_run_map[alias]["run_uid"] = 0
            alias_run_map[alias]["target_acc_epoch"] = 0
            alias_run_map[alias]["target_acc"] = 0

        return x, max_x, y, max_y, None, None
    return get_data_func















