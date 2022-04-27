#!/bin/bash

export entity="hpml-hkbu"
export project="gossipfl"

export cluster_name="scigpu"



export PS_MPI_HOST="scigpu13:33"
export PS_GPU_MAPPINGS="scigpu13:9,8,8,8"

export FEDAVG_MPI_HOST="scigpu13:17"
export FEDAVG_GPU_MAPPINGS="scigpu13:5,4,4,4"

export GOSSIP_MPI_HOST="scigpu13:32"
export GOSSIP_GPU_MAPPINGS="scigpu13:8,8,8,8"


export NWORKERS=32


# export dataset="ptb"
# export model="lstm"
# export task="ptb"
# export trainer_type="lstm"

# export dataset="cifar10"
# export model="resnet20"

# export dataset="shakespeare"
# export model="rnn"


export dataset="mnist"
export model="mnistflnet"


export sched="StepLR"
export lr_decay_rate=0.992

export partition_method='iid'
# export lr=0.1
# lrs=(0.001 0.01 0.01 0.3)
export lr=0.05

export Failure_chance=0.005

# lr=$lr algorithm="PSGD" compression=eftopk compress_ratio=0.01      bash ./launch_mpi_based.sh

# lr=$lr algorithm="FedAvg" compression=no  epochs=1   momentum=0.0          bash ./launch_mpi_based.sh
# lr=$lr algorithm="FedAvg" compression=topk compress_ratio=0.01  epochs=1 \
#      momentum=0.0          bash ./launch_mpi_based.sh
# lr=$lr algorithm="DPSGD"             bash ./launch_mpi_based.sh
# lr=$lr algorithm="DCD_PSGD"          bash ./launch_mpi_based.sh
# lr=$lr algorithm="CHOCO_SGD"         bash ./launch_mpi_based.sh
lr=$lr algorithm="SAPS_FL"           bash ./launch_mpi_based.sh








