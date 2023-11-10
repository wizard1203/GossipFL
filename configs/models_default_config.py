"""
    Note that the epochs
"""


def build_model_default_config(cfg, model):
    if model == "mnistflnet":
        cfg.batch_size = 50
        cfg.lr = 0.05
        cfg.wd = 0.0001
        cfg.momentum = 0.9
        cfg.comm_round = 200
        cfg.epochs = 200
    elif model == "cifar10flnet":
        cfg.batch_size = 100
        cfg.lr = 0.04
        cfg.wd = 0.0001
        cfg.momentum = 0.9
        cfg.comm_round = 400
        cfg.epochs = 400
    elif model == "resnet20":
        cfg.batch_size = 32
        cfg.lr = 0.1
        cfg.wd = 0.0001
        cfg.momentum = 0.9
        cfg.comm_round = 300
        cfg.epochs = 300
    elif model == "resnet56":
        cfg.batch_size = 32
        cfg.lr = 0.1
        cfg.wd = 0.0001
        cfg.momentum = 0.9
        cfg.comm_round = 300
        cfg.epochs = 300
    elif model == "lstm":
        cfg.batch_size = 4
        cfg.lr = 20.0
        cfg.wd = 0.0001
        cfg.momentum = 0.9
        cfg.comm_round = 100
        cfg.epochs = 100
        cfg.lstm_embedding_dim = 1500
    elif model == "lstman4":
        cfg.batch_size = 4
        cfg.lr = 20.0
        cfg.wd = 0.0001
        cfg.momentum = 0.9
        cfg.comm_round = 300
        cfg.epochs = 100
    elif model == "rnn":
        cfg.batch_size = 50
        cfg.lr = 1.0
        cfg.wd = 0.0001
        cfg.momentum = 0.9
        cfg.comm_round = 100
        cfg.epochs = 100
        cfg.lstm_clip_grad = False
        cfg.lstm_embedding_dim = 8
        cfg.lstm_hidden_size = 256
    else:
        pass

    if cfg.algorithm in ["FedAvg", "AFedAvg"]:
        cfg.epochs = 1




