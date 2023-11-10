import os
import logging

from .FederatedEMNIST.data_loader import load_partition_data_federated_emnist
from .fed_cifar100.data_loader import load_partition_data_federated_cifar100
from .fed_shakespeare.data_loader import load_partition_data_federated_shakespeare
from .shakespeare.data_loader import load_partition_data_shakespeare
from .shakespeare.iid_data_loader import load_iid_shakespeare
from .stackoverflow_lr.data_loader import load_partition_data_federated_stackoverflow_lr
from .MNIST.data_loader import load_partition_data_mnist

from .MNIST.iid_data_loader import load_iid_mnist
from .MNIST.centralized_loader import load_centralized_mnist
from .cifar10.iid_data_loader import load_iid_cifar10
from .cifar10.data_loader import load_partition_data_cifar10
from .cifar10.centralized_loader import load_centralized_cifar10
from .cifar100.data_loader import load_partition_data_cifar100
from .cifar100.centralized_loader import load_centralized_cifar100
from .cinic10.data_loader import load_partition_data_cinic10
from .ptb.iid_data_loader import load_iid_ptb
from .FashionMNIST.iid_data_loader import load_iid_FashionMNIST
from .FashionMNIST.data_loader import load_partition_data_fmnist

def load_data(args, dataset_name, **kargs):
    other_params = {}
    if dataset_name == "mnist":
        if args.partition_method == 'iid':
            train_data_num, test_data_num, train_data_global, test_data_global, \
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
            class_num = load_iid_mnist(args.dataset, args.data_dir, args.partition_method,
                    args.partition_alpha, args.client_num_in_total, args.batch_size, args.client_index)
        else:
            logging.info("load_data. dataset_name = %s" % dataset_name)
            # client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
            # train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
            # class_num = load_partition_data_mnist(args.batch_size)
            train_data_num, test_data_num, train_data_global, test_data_global, \
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
            class_num = load_partition_data_mnist(args.dataset, args.data_dir, args.partition_method,
                                    args.partition_alpha, args.client_num_in_total, args.batch_size, args)
            """
            For shallow NN or linear models, 
            we uniformly sample a fraction of clients each round (as the original FedAvg paper)
            """
            # args.client_num_in_total = client_num

    elif dataset_name == "femnist":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_emnist(args.dataset, args.data_dir)
        args.client_num_in_total = client_num
    elif dataset_name == "shakespeare":
        if args.partition_method == 'iid':
            logging.info("load_data. dataset_name = %s" % dataset_name)
            client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
            class_num = load_iid_shakespeare(args.data_dir,
                                            args.batch_size, args.client_index, args)
            args.client_num_in_total = client_num
        else:
            logging.info("load_data. dataset_name = %s" % dataset_name)
            client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
            class_num = load_partition_data_shakespeare(args.batch_size)
            args.client_num_in_total = client_num
    elif dataset_name == "fed_shakespeare":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_shakespeare(args.dataset, args.data_dir)
        args.client_num_in_total = client_num

    elif dataset_name == "fed_cifar100":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_cifar100(args.dataset, args.data_dir)
        args.client_num_in_total = client_num
    elif dataset_name == "stackoverflow_lr":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_stackoverflow_lr(args.dataset, args.data_dir)
        args.client_num_in_total = client_num
    elif dataset_name == "stackoverflow_nwp":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_stackoverflow_nwp(args.dataset, args.data_dir)
        args.client_num_in_total = client_num

    elif dataset_name == "cifar10" and args.partition_method == 'iid':
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_iid_cifar10(args.dataset, args.data_dir, args.partition_method,
                args.partition_alpha, args.client_num_in_total, args.batch_size, args.client_index, args)
    elif dataset_name == "fmnist":
        if args.partition_method == 'iid':
            train_data_num, test_data_num, train_data_global, test_data_global, \
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
            class_num = load_iid_FashionMNIST(args.dataset, args.data_dir, args.partition_method,
                    args.partition_alpha, args.client_num_in_total, args.batch_size, args.client_index, args)
        else:
            train_data_num, test_data_num, train_data_global, test_data_global, \
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
            class_num = load_partition_data_fmnist(args.dataset, args.data_dir, args.partition_method,
                                    args.partition_alpha, args.client_num_in_total, args.batch_size, args)
    elif dataset_name == "ptb":
        if args.partition_method == 'iid':
            train_data_num, test_data_num, train_data_global, test_data_global, \
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
            class_num, other_params = load_iid_ptb(args.dataset, args.data_dir, args.partition_method,
                    args.partition_alpha, args.client_num_in_total, args.batch_size,
                    args.lstm_num_steps, args.client_index)
            logging.info("vocab_size: {}, batch_size :{}, num_steps:{} ".format(
                other_params["vocab_size"], args.batch_size, args.lstm_num_steps))
        else:
            raise NotImplementedError

    else:
        if dataset_name == "cifar10":
            data_loader = load_partition_data_cifar10
        elif dataset_name == "cifar100":
            data_loader = load_partition_data_cifar100
        elif dataset_name == "cinic10":
            data_loader = load_partition_data_cinic10
        else:
            data_loader = load_partition_data_cifar10

        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = data_loader(args.dataset, args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_num_in_total, args.batch_size, args)
    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, other_params]
    return dataset




def load_centralized_data(args, dataset_name, **kargs):
    other_params = {}
    if dataset_name == "mnist":
        train_dl, test_dl, train_data_num, test_data_num, class_num = load_centralized_mnist(args.dataset, args.data_dir, args.batch_size, args, **kargs)
    elif dataset_name == "cifar10":
        train_dl, test_dl, train_data_num, test_data_num, class_num = load_centralized_cifar10(args.dataset, args.data_dir, args.batch_size, args, **kargs)
    elif dataset_name == "cifar100":
        train_dl, test_dl, train_data_num, test_data_num, class_num = load_centralized_cifar100(args.dataset, args.data_dir, args.batch_size, args, **kargs)
    else:
        raise NotImplementedError

    return train_dl, test_dl, train_data_num, test_data_num, class_num, other_params




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    