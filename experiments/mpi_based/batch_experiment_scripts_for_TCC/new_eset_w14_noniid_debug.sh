#!/bin/bash

export entity="hpml-hkbu"
export project="gossipfl"

export cluster_name="esetstore"


# export PS_MPI_HOST="gpu13:9,gpu14:8,gpu15:8,gpu16:8"
# export PS_GPU_MAPPINGS="gpu13:3,2,2,2;gpu14:2,2,2,2;gpu15:2,2,2,2;gpu16:2,2,2,2"
export PS_MPI_HOST="gpu9:9,gpu10:8,gpu11:8,gpu12:8"
export PS_GPU_MAPPINGS="gpu9:3,2,2,2;gpu10:2,2,2,2;gpu11:2,2,2,2;gpu12:2,2,2,2"

# Only use half of workers to train
# export FEDAVG_MPI_HOST="gpu13:5,gpu14:4,gpu15:4,gpu16:4"
# export FEDAVG_GPU_MAPPINGS="gpu13:2,1,1,1;gpu14:1,1,1,1;gpu15:1,1,1,1;gpu16:1,1,1,1"
export FEDAVG_MPI_HOST="gpu9:5,gpu10:4,gpu11:4,gpu12:4"
export FEDAVG_GPU_MAPPINGS="gpu9:2,1,1,1;gpu10:1,1,1,1;gpu11:1,1,1,1;gpu12:1,1,1,1"

# export GOSSIP_MPI_HOST="gpu13:8,gpu14:8,gpu15:8,gpu16:8"
# export GOSSIP_GPU_MAPPINGS="gpu13:2,2,2,2;gpu14:2,2,2,2;gpu15:2,2,2,2;gpu16:2,2,2,2"
export GOSSIP_MPI_HOST="gpu9:8,gpu10:8,gpu11:8,gpu12:8"
export GOSSIP_GPU_MAPPINGS="gpu9:2,2,2,2;gpu10:2,2,2,2;gpu11:2,2,2,2;gpu12:2,2,2,2"


export NWORKERS=32


# export dataset="ptb"
# export model="lstm"
# export task="ptb"
# export trainer_type="lstm"

export dataset="cifar10"
export model="resnet20"

#export dataset="shakespeare"
#export model="rnn"


export sched="StepLR"
export lr_decay_rate=0.992

export partition_method='iid'
export lr=0.1
# lrs=(0.001 0.01 0.01 0.3)
lrs=(0.03 0.3)


#lr=0.1 algorithm="PSGD" compression=no                      bash ./launch_mpi_based.sh

# export level=DEBUG

lr=0.1 algorithm="DPSGD"             bash ./launch_mpi_based.sh

# for lr in "${lrs[@]}"
# do
#     lr=$lr algorithm="PSGD" compression=no                      bash ./launch_mpi_based.sh
#     lr=$lr algorithm="PSGD" compression=topk compress_ratio=0.001   bash ./launch_mpi_based.sh
#     lr=$lr algorithm="PSGD" compression=topk compress_ratio=0.01   bash ./launch_mpi_based.sh
#     lr=$lr algorithm="PSGD" compression=topk compress_ratio=0.1   bash ./launch_mpi_based.sh
#     lr=$lr algorithm="PSGD" compression=eftopk compress_ratio=0.001      bash ./launch_mpi_based.sh
#     lr=$lr algorithm="PSGD" compression=eftopk compress_ratio=0.01      bash ./launch_mpi_based.sh
#     lr=$lr algorithm="PSGD" compression=eftopk compress_ratio=0.1      bash ./launch_mpi_based.sh
#     lr=$lr algorithm="PSGD" compression=qsgd quantize_level=2      bash ./launch_mpi_based.sh
#     lr=$lr algorithm="PSGD" compression=qsgd quantize_level=16      bash ./launch_mpi_based.sh
# 
# 
#     lr=$lr algorithm="APSGD"    bash ./launch_mpi_based.sh
#     lr=$lr algorithm="Local_PSGD"  local_round_num=2   bash ./launch_mpi_based.sh
#     lr=$lr algorithm="Local_PSGD"  local_round_num=4   bash ./launch_mpi_based.sh
#     lr=$lr algorithm="Local_PSGD"  local_round_num=8   bash ./launch_mpi_based.sh
#     lr=$lr algorithm="Local_PSGD"  local_round_num=16  bash ./launch_mpi_based.sh
#     lr=$lr algorithm="Local_PSGD"  local_round_num=2 compression=topk compress_ratio=0.01  bash ./launch_mpi_based.sh
#     lr=$lr algorithm="Local_PSGD"  local_round_num=4 compression=topk compress_ratio=0.01  bash ./launch_mpi_based.sh
#     lr=$lr algorithm="Local_PSGD"  local_round_num=8  compression=topk compress_ratio=0.01 bash ./launch_mpi_based.sh
#     lr=$lr algorithm="Local_PSGD"  local_round_num=16 compression=topk compress_ratio=0.01  bash ./launch_mpi_based.sh
# 
#     lr=$lr algorithm="FedAvg" compression=no  epochs=1   momentum=0.0          bash ./launch_mpi_based.sh
#     lr=$lr algorithm="FedAvg" compression=topk compress_ratio=0.25  epochs=1 \
#          momentum=0.0          bash ./launch_mpi_based.sh
# 
#     lr=$lr algorithm="DPSGD"             bash ./launch_mpi_based.sh
#     lr=$lr algorithm="DCD_PSGD"          bash ./launch_mpi_based.sh
#     lr=$lr algorithm="CHOCO_SGD"         bash ./launch_mpi_based.sh
#     lr=$lr algorithm="SAPS_FL"           bash ./launch_mpi_based.sh
# 
#     # lr=$lr algorithm="FedAvg" compression=no  epochs=1   momentum=0.0          bash ./launch_mpi_based.sh
#     # lr=$lr algorithm="FedAvg" compression=topk compress_ratio=0.01  epochs=1 \
#     #      momentum=0.0          bash ./launch_mpi_based.sh
# 
# done
# 
# 





