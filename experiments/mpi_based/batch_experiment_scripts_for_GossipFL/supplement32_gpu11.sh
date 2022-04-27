CLUSTER_NAME="gpuhome"
# MPIRUN="/home/esetstore/.local/openmpi-4.0.1/bin/mpirun"
# MPIRUN="/home/comp/zhtang/openmpi3.1.1/bin/mpirun"
MPIRUN="/home/comp/zhtang/openmpi4.0.1/bin/mpirun"
# PYTHON=/home/esetstore/zhtang/py36/bin/python
PYTHON="/home/comp/zhtang/py36/bin/python"
# MPI_ARGS=" "
# MPI_ARGS=" --prefix /home/comp/zhtang/openmpi3.1.1 \
#     -mca pml ob1 -mca btl tcp,self \
#     -mca btl_tcp_if_include bond0,eno1,ens1f0np0,ens1f1np1,192.168.0.11/24 \
# "
MPI_ARGS=" --prefix /home/comp/zhtang/openmpi4.0.1 \
    -mca pml ob1 -mca btl tcp,self \
    -mca btl_tcp_if_include bond0 \
    -x LD_LIBRARY_PATH \
"
    # -x NCCL_DEBUG=INFO \
    # -x NCCL_IB_DISABLE=1 \

# bond0 eno1   ens1f0np0 ens1f1np1

# DATASET="cifar10"
# MODEL="resnet20"
# DATASET="cifar10"
# MODEL="cifar10flnet"
DATASET="mnist"
MODEL="mnistflnet"

# PS_MPI_HOST_32="gpu21:33"
# PS_GPU_MAPPINGS_32="gpu21:9,8,8,8"

# FEDAVG_MPI_HOST_32="gpu11:17"
# FEDAVG_GPU_MAPPINGS_32="gpu11:5,4,4,4"

# GOSSIP_MPI_HOST_32="gpu11:32"
# GOSSIP_GPU_MAPPINGS_32="gpu11:8,8,8,8"

PS_MPI_HOST_32="gpu11:17,gpu12:16"
PS_GPU_MAPPINGS_32="gpu11:5,4,4,4;gpu12:4,4,4,4"

# PS_MPI_HOST_32="gpu23:17,gpu24:16"
# PS_GPU_MAPPINGS_32="gpu23:5,4,4,4;gpu24:4,4,4,4"

# PS_MPI_HOST_32="gpu16:17,gpu17:16"
# PS_GPU_MAPPINGS_32="gpu16:5,4,4,4;gpu17:4,4,4,4"

# PS_MPI_HOST_32="gpu20:17,gpu21:16"
# PS_GPU_MAPPINGS_32="gpu20:5,4,4,4;gpu21:4,4,4,4"


FEDAVG_MPI_HOST="gpu11:9,gpu12:8"
FEDAVG_GPU_MAPPINGS="gpu11:3,2,2,2;gpu12:2,2,2,2"

GOSSIP_MPI_HOST="gpu11:16,gpu12:16"
GOSSIP_GPU_MAPPINGS="gpu11:4,4,4,4;gpu12:4,4,4,4"

#MODELS=("mnistflnet" "cifar10flnet" "resnet20")
#batch_size=""
#lr=""


# FedAvg
$MPIRUN $MPI_ARGS -np 17 -host $FEDAVG_MPI_HOST \
    $PYTHON ./main.py \
    gpu_util_parse $FEDAVG_GPU_MAPPINGS \
    client_num_per_round 16 client_num_in_total 32 \
    dataset $DATASET  cluster_name $CLUSTER_NAME \
    partition_method iid \
    entity hpml-hkbu project gossipfl algorithm FedAvg psgd_exchange model if_get_diff False \
    model $MODEL \
    client_optimizer sgd momentum 0.0 nesterov False \
    sched StepLR lr_decay_rate 0.992

# FedAvg sparse
$MPIRUN $MPI_ARGS -np 17 -host $FEDAVG_MPI_HOST \
    $PYTHON ./main.py \
    gpu_util_parse $FEDAVG_GPU_MAPPINGS \
    client_num_per_round 16 client_num_in_total 32 \
    dataset $DATASET  cluster_name $CLUSTER_NAME \
    partition_method iid \
    entity hpml-hkbu project gossipfl algorithm FedAvg psgd_exchange model if_get_diff True \
    model $MODEL \
    client_optimizer sgd momentum  0.0 nesterov False \
    sched StepLR lr_decay_rate 0.992 \
    compression topk compress_ratio 0.25 quantize_level 32  is_biased  0


# PSGD
# $MPIRUN $MPI_ARGS -np 33 -host $PS_MPI_HOST_32 \
#     $PYTHON ./main.py \
#     gpu_util_parse $PS_GPU_MAPPINGS_32 \
#     client_num_per_round 32 client_num_in_total 32 \
#     dataset $DATASET  cluster_name $CLUSTER_NAME \
#     partition_method iid \
#     entity hpml-hkbu project gossipfl algorithm PSGD psgd_exchange grad \
#     model $MODEL \
#     client_optimizer sgd \
#     sched StepLR lr_decay_rate 0.992 \
#     compression topk compress_ratio 0.01 quantize_level 32  is_biased  0



# # DCD_PSGD
# $MPIRUN $MPI_ARGS -np 32 -host $GOSSIP_MPI_HOST_32 \
#     $PYTHON ./main.py \
#     gpu_util_parse $GOSSIP_GPU_MAPPINGS_32 \
#     client_num_per_round 32 client_num_in_total 32 \
#     dataset $DATASET  cluster_name $CLUSTER_NAME \
#     partition_method iid \
#     entity hpml-hkbu project gossipfl algorithm DCD_PSGD  \
#     model $MODEL \
#     client_optimizer sgd \
#     sched StepLR lr_decay_rate 0.992 \
#     compression topk compress_ratio 0.25 quantize_level 32  is_biased  0

# # CHOCO_SGD
# $MPIRUN $MPI_ARGS -np 32 -host $GOSSIP_MPI_HOST_32 \
#     $PYTHON ./main.py \
#     gpu_util_parse $GOSSIP_GPU_MAPPINGS_32 \
#     client_num_per_round 32 client_num_in_total 32 \
#     dataset $DATASET  cluster_name $CLUSTER_NAME \
#     partition_method iid \
#     entity hpml-hkbu project gossipfl algorithm CHOCO_SGD  \
#     model $MODEL \
#     client_optimizer sgd \
#     sched StepLR lr_decay_rate 0.992 \
#     compression topk compress_ratio 0.01 quantize_level 32  is_biased  0 \
#     consensus_stepsize  0.5






