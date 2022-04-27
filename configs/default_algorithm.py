
def build_algorithm_default_config(
    cfg=None, algorithm="PSGD"
):
    if algorithm == "APSGD":
        cfg.algorithm = "APSGD"

        # There is something could be adjusted
        cfg.psgd_exchange = "model"
        cfg.compression = "no"

        # There is something could be adjusted
        cfg.if_get_diff = True
        cfg.psgd_grad_sum = False
        cfg.psgd_grad_debug = False

        cfg.client_optimizer = "sgd"
        cfg.server_optimizer = "no"

    elif algorithm == "CHOCO_SGD":
        cfg.algorithm = "CHOCO_SGD"
        cfg.consensus_stepsize = -0.5

        # There is something could be adjusted
        cfg.psgd_exchange = "model"
        cfg.compression = "no"
        cfg.compress_ratio = -0.01

        # There is something could be adjusted
        cfg.if_get_diff = True
        cfg.psgd_grad_sum = False
        cfg.psgd_grad_debug = False

        cfg.client_optimizer = "sgd"
        cfg.server_optimizer = "no"

    elif algorithm == "DCD_PSGD":
        cfg.algorithm = "DCD_PSGD"
        # There is something could be adjusted
        cfg.psgd_exchange = "model"
        cfg.compression = "topk"
        cfg.compress_ratio = -0.25

        # There is something could be adjusted
        cfg.if_get_diff = False
        cfg.psgd_grad_sum = False
        cfg.psgd_grad_debug = False

        cfg.client_optimizer = "sgd"
        cfg.server_optimizer = "no"

    elif algorithm == "DPSGD":
        cfg.algorithm = "DPSGD"
        # There is something could be adjusted
        cfg.psgd_exchange = "model"
        cfg.compression = "no"

        # There is something could be adjusted
        cfg.if_get_diff = False
        cfg.psgd_grad_sum = False
        cfg.psgd_grad_debug = False

        cfg.client_optimizer = "sgd"
        cfg.server_optimizer = "no"

    elif algorithm == "FedAvg":
        cfg.algorithm = "FedAvg"
        # There is something could be adjusted
        cfg.psgd_exchange = "model"
        cfg.compression = "no"

        # There is something could be adjusted
        cfg.if_get_diff = False
        cfg.psgd_grad_sum = False
        cfg.psgd_grad_debug = False

        cfg.client_optimizer = "sgd"
        cfg.server_optimizer = "no"

        cfg.global_epochs_per_round = 1

    elif algorithm == "FedSGD":
        cfg.algorithm = "FedSGD"
        # There is something could be adjusted
        cfg.psgd_exchange = "grad"
        cfg.compression = "no"

        # There is something could be adjusted
        cfg.if_get_diff = False
        cfg.psgd_grad_sum = False
        cfg.psgd_grad_debug = False

        cfg.client_optimizer = "no"
        cfg.server_optimizer = "sgd"

    elif algorithm == "Local_PSGD":
        cfg.algorithm = "Local_PSGD"
        # There is something could be adjusted
        cfg.psgd_exchange = "model"
        cfg.compression = "no"

        # There is something could be adjusted
        cfg.if_get_diff = False
        cfg.psgd_grad_sum = False
        cfg.psgd_grad_debug = False

        cfg.client_optimizer = "sgd"
        cfg.server_optimizer = "no"

        cfg.local_round_num = 4

    elif algorithm == "PSGD":
        cfg.algorithm = "PSGD"
        # There is something could be adjusted
        cfg.psgd_exchange = "grad"
        cfg.compression = "no"

        # There is something could be adjusted
        cfg.if_get_diff = False
        cfg.psgd_grad_sum = False
        cfg.psgd_grad_debug = False

        cfg.client_optimizer = "sgd"
        cfg.server_optimizer = "no"

    elif algorithm == "SAPS_FL":
        cfg.algorithm = "SAPS_FL"
        # There is something could be adjusted
        cfg.psgd_exchange = "grad"
        cfg.compression = "randomk"
        cfg.compress_ratio = 0.01

        # There is something could be adjusted
        cfg.if_get_diff = False
        cfg.psgd_grad_sum = False
        cfg.psgd_grad_debug = False

        cfg.client_optimizer = "sgd"
        cfg.server_optimizer = "no"



