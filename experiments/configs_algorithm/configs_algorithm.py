
def algorithm_config_for_launch(
    PS_PROCESS, PS_MPI_HOST, PS_GPU_MAPPINGS, PS_CLIENT_NUM,
    GOSSIP_PROCESS, GOSSIP_MPI_HOST, GOSSIP_GPU_MAPPINGS, GOSSIP_CLIENT_NUM,
    FEDAVG_PROCESS, FEDAVG_MPI_HOST, FEDAVG_GPU_MAPPINGS, FEDAVG_CLIENT_NUM, FEDAVG_CLIENT_TOTAL,
    MPI_PROCESS=None, MPI_HOST=None,
    algorithm="PSGD"
):

    if algorithm == "APSGD":
        MPI_PROCESS = PS_PROCESS if MPI_PROCESS is None else MPI_PROCESS
        MPI_HOST = PS_MPI_HOST if MPI_HOST is None else MPI_HOST

        gpu_util_parse = PS_GPU_MAPPINGS
        client_num_per_round = PS_CLIENT_NUM
        client_num_in_total = PS_CLIENT_NUM

    elif algorithm == "CHOCO_SGD":
        MPI_PROCESS = GOSSIP_PROCESS if MPI_PROCESS is None else MPI_PROCESS
        MPI_HOST = GOSSIP_MPI_HOST if MPI_HOST is None else MPI_HOST

        gpu_util_parse = GOSSIP_GPU_MAPPINGS
        client_num_per_round = GOSSIP_CLIENT_NUM
        client_num_in_total = GOSSIP_CLIENT_NUM

    elif algorithm == "CHOCO_SGD":
        MPI_PROCESS = GOSSIP_PROCESS if MPI_PROCESS is None else MPI_PROCESS
        MPI_HOST = GOSSIP_MPI_HOST if MPI_HOST is None else MPI_HOST

        gpu_util_parse = GOSSIP_GPU_MAPPINGS
        client_num_per_round = GOSSIP_CLIENT_NUM
        client_num_in_total = GOSSIP_CLIENT_NUM

    elif algorithm == "DPSGD":
        MPI_PROCESS = GOSSIP_PROCESS if MPI_PROCESS is None else MPI_PROCESS
        MPI_HOST = GOSSIP_MPI_HOST if MPI_HOST is None else MPI_HOST

        gpu_util_parse = GOSSIP_GPU_MAPPINGS
        client_num_per_round = GOSSIP_CLIENT_NUM
        client_num_in_total = GOSSIP_CLIENT_NUM

    elif algorithm == "FedAvg":
        MPI_PROCESS = FEDAVG_PROCESS if MPI_PROCESS is None else MPI_PROCESS
        MPI_HOST = FEDAVG_MPI_HOST if MPI_HOST is None else MPI_HOST

        gpu_util_parse = FEDAVG_GPU_MAPPINGS
        client_num_per_round = FEDAVG_CLIENT_NUM
        client_num_in_total = FEDAVG_CLIENT_TOTAL

    elif algorithm == "FedSGD":
        MPI_PROCESS = PS_PROCESS if MPI_PROCESS is None else MPI_PROCESS
        MPI_HOST = PS_MPI_HOST if MPI_HOST is None else MPI_HOST

        gpu_util_parse = PS_GPU_MAPPINGS
        client_num_per_round = FEDAVG_CLIENT_NUM
        client_num_in_total = FEDAVG_CLIENT_TOTAL

    elif algorithm == "Local_PSGD":
        MPI_PROCESS = PS_PROCESS if MPI_PROCESS is None else MPI_PROCESS
        MPI_HOST = PS_MPI_HOST if MPI_HOST is None else MPI_HOST

        gpu_util_parse = PS_GPU_MAPPINGS
        client_num_per_round = PS_CLIENT_NUM
        client_num_in_total = PS_CLIENT_NUM

    elif algorithm == "PSGD":
        MPI_PROCESS = PS_PROCESS if MPI_PROCESS is None else MPI_PROCESS
        MPI_HOST = PS_MPI_HOST if MPI_HOST is None else MPI_HOST

        gpu_util_parse = PS_GPU_MAPPINGS
        client_num_per_round = PS_CLIENT_NUM
        client_num_in_total = PS_CLIENT_NUM

    elif algorithm == "SAPS_FL":
        MPI_PROCESS = GOSSIP_PROCESS if MPI_PROCESS is None else MPI_PROCESS
        MPI_HOST = GOSSIP_MPI_HOST if MPI_HOST is None else MPI_HOST

        gpu_util_parse = GOSSIP_GPU_MAPPINGS
        client_num_per_round = GOSSIP_CLIENT_NUM
        client_num_in_total = GOSSIP_CLIENT_NUM
    return MPI_PROCESS, MPI_HOST, gpu_util_parse, client_num_per_round, client_num_in_total












