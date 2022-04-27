import logging

def find_data_path(dataset=None, cluster_name=None):
    cluster_data_path_dict = {
        'scigpu': scigpu_data_path,
        'DAAI': DAAI_data_path,
        'gpuhome': gpuhome_data_path,
        't716': t716_data_path,
        'esetstore': esetstore_data_path,
    }
    logging.info("Loading dataset {} on cluster {}".format(dataset, cluster_name))
    return cluster_data_path_dict[cluster_name](dataset)



def scigpu_data_path(dataset):
    data_path_dict = {
        'ILSVRC2012-100': '/home/datasets/imagenet/ILSVRC2012_dataset',
        'ILSVRC2012': '/home/datasets/imagenet/ILSVRC2012_dataset',
        'gld23k': '~/datasets/landmarks',
        'cifar10': '~/datasets/cifar10',
        'mnist': '~/datasets',
        'ptb': '/home/comp/20481896/datasets/PennTreeBank',
        'shakespeare': '/home/comp/20481896/datasets/shakespeare',
    }
    if dataset in data_path_dict: 
        return data_path_dict[dataset]
    else:
        return None


def DAAI_data_path(dataset):
    data_path_dict = {
        'ILSVRC2012-100': '/home/datasets/ILSVRC2012_dataset',
        'ILSVRC2012': '/home/datasets/ILSVRC2012_dataset',
        'gld23k': '/home/datasets/landmarks',
        'cifar10': '/home/datasets/cifar10',
        'ptb': '/home/datasets/PennTreeBank',
        'shakespeare': '/home/datasets/shakespeare',
    }
    if dataset in data_path_dict: 
        return data_path_dict[dataset]
    else:
        return None


def gpuhome_data_path(dataset):
    data_path_dict = {
        'cifar10': '/home/comp/zhtang/dc2-p2p-dl2/data',
        'mnist': '/home/comp/zhtang/dc2-p2p-dl2/data',
        'ptb': '/home/comp/zhtang/data/PennTreeBank',
    }
    if dataset in data_path_dict: 
        return data_path_dict[dataset]
    else:
        return None

def t716_data_path(dataset):
    data_path_dict = {
        'ILSVRC2012-100': '/nfs_home/datasets/ILSVRC2012',
        'ILSVRC2012': '/nfs_home/datasets/ILSVRC2012',
        'gld23k': '/nfs_home/datasets/landmarks',
        'cifar10': '/nfs_home/datasets/cifar10',
        'mnist': '/nfs_home/datasets/mnist',
    }
    if dataset in data_path_dict: 
        return data_path_dict[dataset]
    else:
        return None

def esetstore_data_path(dataset):
    data_path_dict = {
        'ILSVRC2012-100': '/home/esetstore/dataset/ILSVRC2012_dataset',
        'ILSVRC2012': '/home/esetstore/dataset/ILSVRC2012_dataset',
        'gld23k': '/home/esetstore/dataset/landmarks',
        'cifar10': '/home/esetstore/dataset/cifar10',
        'mnist': '/home/esetstore/dataset',
        'ptb': '/home/esetstore/repos/p2p/data/PennTreeBank',
        'shakespeare': '/home/esetstore/dataset/shakespeare',
    }
    if dataset in data_path_dict: 
        return data_path_dict[dataset]
    else:
        return None

