#!/bin/bash

export entity="zhtang"
export project="test"

export cluster_name="scigpu"
export gpu_index=3
export client_num_per_round=5
export client_num_in_total=10


# export dataset="cifar10"
export dataset="mnist"

# model="resnet20" bash ./model_scan/launch_model_scan.sh


model="vgg-9" bash ./model_scan/launch_model_scan.sh

# model="resnet18_v2" bash ./model_scan/launch_model_scan.sh














