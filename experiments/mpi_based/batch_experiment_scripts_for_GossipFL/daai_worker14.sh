#!/bin/bash

# slurm batch script
#SBATCH -o logs/slurm.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=15
#SBATCH -w hkbugpusrv06

CLUSTER_NAME="DAAI"
# MPIRUN="/home/esetstore/.local/openmpi-4.0.1/bin/mpirun"
MPIRUN=mpirun
# PYTHON=/home/esetstore/zhtang/py36/bin/python
# PYTHON="/home/comp/20481896/anaconda3/envs/py36/bin/python"
PYTHON="/home/comp/20481896/py36/bin/python"
MPI_ARGS=" "
# MPI_ARGS=" --prefix /usr/local/openmpi/openmpi-4.0.1 \
#     -mca pml ob1 -mca btl ^openib,vader,self \
#     -mca btl_openib_allow_ib 1 \
#     -mca btl_tcp_if_include bond0
#     -mca btl_openib_want_fork_support 1 \
#     -x LD_LIBRARY_PATH \
#     -x NCCL_DEBUG=INFO  \
#     -x NCCL_SOCKET_IFNAME=bond0 \
#     -x NCCL_IB_DISABLE=0 \
# "

PS_MPI_HOST_14="hkbugpusrv06:15"
PS_GPU_MAPPINGS_14="hkbugpusrv06:4,4,4,3"

# Only use half of workers to train
FEDAVG_MPI_HOST_14="hkbugpusrv06:8"
FEDAVG_GPU_MAPPINGS_14="hkbugpusrv06:2,2,2,2"

GOSSIP_MPI_HOST_14="hkbugpusrv06:14"
GOSSIP_GPU_MAPPINGS_14="hkbugpusrv06:4,4,3,3"



#MODELS=("mnistflnet" "cifar10flnet" "resnet20")
#batch_size=""
#lr=""


# PSGD
$MPIRUN $MPI_ARGS -np 15 -host $PS_MPI_HOST_14 \
    $PYTHON ./main.py \
    gpu_util_parse $PS_GPU_MAPPINGS_14 \
    client_num_per_round 14 client_num_in_total 14 \
    dataset cifar10  cluster_name $CLUSTER_NAME \
    partition_method iid \
    entity hpml-hkbu project gossipfl algorithm PSGD psgd_exchange grad \
    model cifar10flnet \
    client_optimizer sgd \
    sched StepLR lr_decay_rate 0.992 lr 0.01 \

