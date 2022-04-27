#!/bin/bash

export entity="hpml-hkbu"
export project="gossipfl"

export cluster_name="esetstore"


# export PS_MPI_HOST="gpu13:9,gpu14:8,gpu15:8,gpu16:8"
# export PS_GPU_MAPPINGS="gpu13:3,2,2,2;gpu14:2,2,2,2;gpu15:2,2,2,2;gpu16:2,2,2,2"
export PS_MPI_HOST="gpu9:13,gpu10:13,gpu11:13,gpu12:12,gpu13:12,gpu14:13,gpu15:13,gpu16:12"
export PS_GPU_MAPPINGS="gpu9:4,3,3,3;gpu10:4,3,3,3;gpu11:4,3,3,3;gpu12:3,3,3,3;gpu13:3,3,3,3;gpu14:4,3,3,3;gpu15:4,3,3,3;gpu16:3,3,3,3"

# Only use half of workers to train
# export FEDAVG_MPI_HOST="gpu13:5,gpu14:4,gpu15:4,gpu16:4"
# export FEDAVG_GPU_MAPPINGS="gpu13:2,1,1,1;gpu14:1,1,1,1;gpu15:1,1,1,1;gpu16:1,1,1,1"
export FEDAVG_MPI_HOST="gpu9:7,gpu10:7,gpu11:7,gpu12:6,gpu13:6,gpu14:6,gpu15:6,gpu16:6"
export FEDAVG_GPU_MAPPINGS="gpu9:2,2,2,1;gpu10:2,2,2,1;gpu11:2,2,2,1;gpu12:2,2,1,1;gpu13:2,2,1,1;gpu14:2,2,1,1;gpu15:2,2,1,1;gpu16:2,2,1,1"

# export GOSSIP_MPI_HOST="gpu13:8,gpu14:8,gpu15:8,gpu16:8"
# export GOSSIP_GPU_MAPPINGS="gpu13:2,2,2,2;gpu14:2,2,2,2;gpu15:2,2,2,2;gpu16:2,2,2,2"
export GOSSIP_MPI_HOST="gpu9:13,gpu10:13,gpu11:12,gpu12:12,gpu13:12,gpu14:13,gpu15:13,gpu16:12"
export GOSSIP_GPU_MAPPINGS="gpu9:4,3,3,3;gpu10:4,3,3,3;gpu11:3,3,3,3;gpu12:3,3,3,3;gpu13:3,3,3,3;gpu14:4,3,3,3;gpu15:4,3,3,3;gpu16:3,3,3,3"


export NWORKERS=100


# export dataset="ptb"
# export model="lstm"
# export task="ptb"
# export trainer_type="lstm"

export dataset="cifar10"
export model="cifar10flnet"

#export dataset="shakespeare"
#export model="rnn"


export sched="StepLR"
export lr_decay_rate=0.992

# export partition_method='iid'
# export partition_method='hetero'
export partition_method='noniid-#label2'


export lr=0.1
# lrs=(0.001 0.01 0.01 0.3)
# lrs=(0.03 0.3)
lrs=(0.1)

for lr in "${lrs[@]}"
do
    lr=$lr partition_alpha=$partition_alpha algorithm="PSGD" bash ./launch_mpi_based.sh
    # lr=$lr partition_alpha=$partition_alpha algorithm="PSGD" compression=eftopk compress_ratio=0.01      bash ./launch_mpi_based.sh
    lr=$lr partition_alpha=$partition_alpha algorithm="FedAvg" compression=no  epochs=1   momentum=0.0          bash ./launch_mpi_based.sh
    # lr=$lr partition_alpha=$partition_alpha algorithm="FedAvg" compression=topk compress_ratio=0.01  epochs=1 \
    #     momentum=0.0          bash ./launch_mpi_based.sh
    # lr=$lr partition_alpha=$partition_alpha algorithm="DPSGD"             bash ./launch_mpi_based.sh
    lr=$lr partition_alpha=$partition_alpha algorithm="SAPS_FL"           bash ./launch_mpi_based.sh
    lr=$lr partition_alpha=$partition_alpha algorithm="CHOCO_SGD"         bash ./launch_mpi_based.sh
    # lr=$lr partition_alpha=$partition_alpha algorithm="DCD_PSGD"          bash ./launch_mpi_based.sh
done



# for lr in "${lrs[@]}"
# do
#      for partition_alpha in "${partition_alpha_list[@]}"
#      do
#           lr=$lr algorithm="PSGD" bash ./launch_mpi_based.sh
#           lr=$lr algorithm="PSGD" compression=eftopk compress_ratio=0.01      bash ./launch_mpi_based.sh
#           lr=$lr algorithm="FedAvg" compression=no  epochs=1   momentum=0.0          bash ./launch_mpi_based.sh
#           lr=$lr algorithm="FedAvg" compression=topk compress_ratio=0.01  epochs=1 \
#                momentum=0.0          bash ./launch_mpi_based.sh
#           lr=$lr algorithm="DPSGD"             bash ./launch_mpi_based.sh
#           lr=$lr partition_alpha=$partition_alpha algorithm="SAPS_FL"           bash ./launch_mpi_based.sh
#           lr=$lr partition_alpha=$partition_alpha algorithm="CHOCO_SGD"         bash ./launch_mpi_based.sh
#           lr=$lr partition_alpha=$partition_alpha algorithm="DCD_PSGD"          bash ./launch_mpi_based.sh
#      done
# done




