CLUSTER_NAME="gpuhome"
MPIRUN="/home/esetstore/.local/openmpi-4.0.1/bin/mpirun"
MPIRUN=mpirun
# PYTHON=/home/esetstore/zhtang/py36/bin/python
PYTHON="/home/comp/zhtang/py36/bin/python"
MPI_ARGS=" "
# DATASET="cifar10"
# MODEL="resnet20"
# DATASET="mnist"
# MODEL="mnistflnet"

PS_MPI_HOST_32="gpu20:33"
PS_GPU_MAPPINGS_32="gpu20:9,8,8,8"

FEDAVG_MPI_HOST_32="gpu20:17"
FEDAVG_GPU_MAPPINGS_32="gpu20:5,4,4,4"

GOSSIP_MPI_HOST_32="gpu20:32"
GOSSIP_GPU_MAPPINGS_32="gpu20:8,8,8,8"

MODELS=("mnistflnet" "cifar10flnet" "resnet20")
batch_size=""
lr=""



for MODEL in "${MODELS[@]}"
do
    if [ "$MODEL" = "mnistflnet" ];then
        DATASET="mnist"
    else
        DATASET="cifar10"
    fi

    $MPIRUN $MPI_ARGS -np 33 -host $PS_MPI_HOST_32 \
        $PYTHON ./main.py \
        gpu_util_parse $PS_GPU_MAPPINGS_32 \
        client_num_per_round 32 client_num_in_total 32 \
        dataset $DATASET  cluster_name $CLUSTER_NAME \
        partition_method iid \
        entity hpml-hkbu project gossipfl algorithm PSGD psgd_exchange grad \
        model $MODEL \
        client_optimizer sgd \
        sched StepLR lr_decay_rate 0.992 \
        compression eftopk compress_ratio 0.01 quantize_level 32  is_biased  0

    # FedAvg
    $MPIRUN $MPI_ARGS -np 17 -host $FEDAVG_MPI_HOST_32 \
        $PYTHON ./main.py \
        gpu_util_parse $FEDAVG_GPU_MAPPINGS_32 \
        client_num_per_round 16 client_num_in_total 32 \
        dataset $DATASET  cluster_name $CLUSTER_NAME \
        partition_method iid \
        entity hpml-hkbu project gossipfl algorithm FedAvg psgd_exchange model if_get_diff False \
        model $MODEL \
        client_optimizer sgd \
        sched StepLR lr_decay_rate 0.992

    # FedAvg sparse
    $MPIRUN $MPI_ARGS -np 17 -host $FEDAVG_MPI_HOST_32 \
        $PYTHON ./main.py \
        gpu_util_parse $FEDAVG_GPU_MAPPINGS_32 \
        client_num_per_round 16 client_num_in_total 32 \
        dataset $DATASET  cluster_name $CLUSTER_NAME \
        partition_method iid \
        entity hpml-hkbu project gossipfl algorithm FedAvg psgd_exchange model if_get_diff True \
        model $MODEL \
        client_optimizer sgd \
        sched StepLR lr_decay_rate 0.992 \
        compression topk compress_ratio 0.25 quantize_level 32  is_biased  0

done




