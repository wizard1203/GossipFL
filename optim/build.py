import torch

from .fedprox import FedProx


"""
    args.opt in 
    ["sgd", "adam"]
    --lr
    --momentum
    --clip-grad # wait to be developed
    --weight-decay, --wd
"""

def create_optimizer(args, model=None, params=None, **kwargs):
    if "role" in kwargs:
        role = kwargs["role"]
    else:
        role = args.role

    if model is not None:
        params_to_optimizer = model.parameters()
    else:
        pass

    # params has higher priority than model
    if params is not None:
        params_to_optimizer = params
    else:
        pass
    assert params_to_optimizer is not None

    if (role == 'server') and (args.algorithm in [
        'FedAvg', 'AFedAvg', 'PSGD', 'APSGD', 'Local_PSGD', 'FedSGD']):
        if args.server_optimizer == "sgd":
            optimizer = torch.optim.SGD(params_to_optimizer,
                lr=args.lr, weight_decay=args.wd, momentum=args.momentum, nesterov=args.nesterov)
        elif args.server_optimizer == "adam":
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params_to_optimizer),
                lr=args.lr, weight_decay=args.wd, amsgrad=True)
        elif args.server_optimizer == "FedProx":
            optimizer = FedProx(params_to_optimizer, lr=args.lr, momentum=args.momentum,
                                mu=args.fedprox_mu, nesterov=args.nesterov, weight_decay=args.wd)
        elif args.server_optimizer == "no":
            optimizer = torch.optim.SGD(params_to_optimizer,
                lr=args.lr, weight_decay=args.wd, momentum=args.momentum, nesterov=args.nesterov)
        else:
            raise NotImplementedError
    else:
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(params_to_optimizer,
                lr=args.lr, weight_decay=args.wd, momentum=args.momentum, nesterov=args.nesterov)
        elif args.client_optimizer == "adam":
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params_to_optimizer),
                lr=args.lr, weight_decay=args.wd, amsgrad=True)
        elif args.client_optimizer == "FedProx":
            optimizer = FedProx(params_to_optimizer, lr=args.lr, momentum=args.momentum,
                                mu=args.fedprox_mu, nesterov=args.nesterov, weight_decay=args.wd)
        elif args.client_optimizer == "no":
            optimizer = torch.optim.SGD(params_to_optimizer,
                lr=args.lr, weight_decay=args.wd, momentum=args.momentum, nesterov=args.nesterov)
        else:
            raise NotImplementedError


    return optimizer







