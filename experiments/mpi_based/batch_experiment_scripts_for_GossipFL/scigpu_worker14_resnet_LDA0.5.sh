#!/bin/bash

export entity="hpml-hkbu"
export project="gossipfl"

export cluster_name="scigpu"




# export PS_MPI_HOST="scigpu12:15"
# export PS_GPU_MAPPINGS="scigpu12:4,4,4,3"

# export FEDAVG_MPI_HOST="scigpu12:8"
# export FEDAVG_GPU_MAPPINGS="scigpu12:2,2,2,2"

# export GOSSIP_MPI_HOST="scigpu12:14"
# # export GOSSIP_GPU_MAPPINGS="scigpu12:4,4,3,3"
# export GOSSIP_GPU_MAPPINGS="scigpu12:4,5,5,0"



export PS_MPI_HOST="scigpu10:15"
export PS_GPU_MAPPINGS="scigpu10:4,4,4,3"

export FEDAVG_MPI_HOST="scigpu10:8"
export FEDAVG_GPU_MAPPINGS="scigpu10:2,2,2,2"

export GOSSIP_MPI_HOST="scigpu10:14"
# export GOSSIP_GPU_MAPPINGS="scigpu12:4,4,3,3"
export GOSSIP_GPU_MAPPINGS="scigpu10:4,4,3,3"



export NWORKERS=14


# export dataset="ptb"
# export model="lstm"
# export task="ptb"
# export trainer_type="lstm"


# export dataset="mnist"
# export model="mnistflnet"

export dataset="cifar10"
# export model="cifar10flnet"
# lrs=(0.04)

export model="resnet20"



# export dataset="shakespeare"
# export model="rnn"


# export dataset="mnist"
# export model="mnistflnet"


export sched="StepLR"
export lr_decay_rate=0.992

# export partition_method='iid'
export partition_method='hetero'
export partition_alpha=0.5
# export partition_alpha=10.0
# partition_alpha_list=(1.0 10.0)
# partition_alpha_list=(10.0)

# export partition_method='noniid-#label2'
# export partition_alpha=0.5


# export lr=0.1
# lrs=(0.01 0.01 0.3)
# lrs=(0.03 0.1)
# lrs=(0.03)
lrs=(0.1)

# export lr=0.1

# export Failure_chance=0.005

# lr=$lr algorithm="PSGD" bash ./launch_mpi_based.sh
# lr=$lr algorithm="PSGD" compression=eftopk compress_ratio=0.01      bash ./launch_mpi_based.sh

# lr=$lr algorithm="FedAvg" compression=no  epochs=1   momentum=0.0          bash ./launch_mpi_based.sh
# lr=$lr algorithm="FedAvg" compression=topk compress_ratio=0.01  epochs=1 \
#      momentum=0.0          bash ./launch_mpi_based.sh
# lr=$lr algorithm="DPSGD"             bash ./launch_mpi_based.sh

# lr=$lr algorithm="SAPS_FL"           bash ./launch_mpi_based.sh
# lr=$lr algorithm="CHOCO_SGD"         bash ./launch_mpi_based.sh
# lr=$lr algorithm="DCD_PSGD"          bash ./launch_mpi_based.sh

for lr in "${lrs[@]}"
do
     # lr=$lr partition_alpha=$partition_alpha algorithm="PSGD" bash ./launch_mpi_based.sh
     # lr=$lr partition_alpha=$partition_alpha algorithm="PSGD" compression=eftopk compress_ratio=0.01      bash ./launch_mpi_based.sh
     # lr=$lr partition_alpha=$partition_alpha algorithm="FedAvg" compression=no  epochs=1   momentum=0.0          bash ./launch_mpi_based.sh
     # lr=$lr partition_alpha=$partition_alpha algorithm="FedAvg" compression=topk compress_ratio=0.25  epochs=1 \
     #      momentum=0.0          bash ./launch_mpi_based.sh
     # lr=$lr partition_alpha=$partition_alpha algorithm="DPSGD"             bash ./launch_mpi_based.sh
     lr=$lr partition_alpha=$partition_alpha algorithm="SAPS_FL" compress_ratio=0.25 \
               bash ./launch_mpi_based.sh
     lr=$lr partition_alpha=$partition_alpha algorithm="CHOCO_SGD"  compress_ratio=0.25 \
               bash ./launch_mpi_based.sh
     # lr=$lr partition_alpha=$partition_alpha algorithm="DCD_PSGD"          bash ./launch_mpi_based.sh

done


# for lr in "${lrs[@]}"
# do
#      lr=$lr algorithm="SAPS_FL"           bash ./launch_mpi_based.sh
#      lr=$lr algorithm="CHOCO_SGD"         bash ./launch_mpi_based.sh
# done





