#!/bin/bash

######################################  mpi_based launch shell
cluster_name="${cluster_name:-localhost}"
model="${model:-resnet20}"
dataset="${dataset:-cifar10}"

dir_name=$(dirname "$PWD")

source ${dir_name}/experiments/configs_system/$cluster_name.conf
source ${dir_name}/experiments/configs_model/$model.conf
source ${dir_name}/experiments/main_args.conf


extra_args="${extra_args:- }"
main_args="${main_args:-  }"


PYTHON="${PYTHON:-python}"

export WANDB_CONSOLE=off

echo $main_args
echo $extra_args
$PYTHON ./model_scan/main.py $extra_args $main_args \









