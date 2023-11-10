#!/bin/bash

######################################  mpi_based launch shell
cluster_name="${cluster_name:-localhost}"
model="${model:-resnet20}"
dataset="${dataset:-cifar10}"
algorithm="${algorithm:-PSGD}"

dir_name=$(dirname "$PWD")

source ${dir_name}/configs_system/$cluster_name.conf
source ${dir_name}/configs_model/$model.conf
source ${dir_name}/configs_algorithm/$algorithm.conf
source ${dir_name}/main_args.conf

main_args="${main_args:-  }"

# MPIRUN="/home/esetstore/.local/openmpi-4.0.1/bin/mpirun"
MPIRUN="${MPIRUN:-mpirun}"
PYTHON="${PYTHON:-python}"
MPI_ARGS="${MPI_ARGS:- }"

MPI_PROCESS="${MPI_PROCESS:-$PS_PROCESS}"
MPI_HOST="${MPI_HOST:-$PS_MPI_HOST}"

export WANDB_CONSOLE=off

echo ${MPIRUN},${MPI_ARGS},${MPI_PROCESS},${MPI_HOST},
echo $main_args
$MPIRUN $MPI_ARGS -np $MPI_PROCESS -host $MPI_HOST \
    $PYTHON ./main.py $main_args \









