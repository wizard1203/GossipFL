import json
import os
import logging

import numpy as np
import torch
import torch.utils.data as data

from .language_utils import word_to_indices, VOCAB_SIZE, \
    letter_to_index
from . import utils


def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(sorted(train_data.keys()))

    return clients, groups, train_data, test_data


def process_x(raw_x_batch):
    x_batch = [word_to_indices(word) for word in raw_x_batch]
    return x_batch


def process_y(raw_y_batch):
    y_batch = [letter_to_index(c) for c in raw_y_batch]
    return y_batch


def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        batched_x = torch.from_numpy(np.asarray(process_x(batched_x)))
        batched_y = torch.from_numpy(np.asarray(process_y(batched_y)))
        batch_data.append((batched_x, batched_y))
    return batch_data


def load_iid_shakespeare(data_dir="../../../data/shakespeare/", batch_size=20, rank=0, args=None):
    # train_path = "../../../data/shakespeare/train"
    # test_path = "../../../data/shakespeare/test"
    train_path = os.path.join(data_dir, "train")
    test_path = os.path.join(data_dir, "test")
    users, groups, train_data, test_data = read_data(train_path, test_path)

    if len(groups) == 0:
        groups = [None for _ in users]
    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()
    # client_index = 0

    all_train_data_x = []
    all_train_data_y = []
    all_test_data_x = []
    all_test_data_y = []

    for u, g in zip(users, groups):
        user_train_data_num = len(train_data[u]['x'])
        user_test_data_num = len(test_data[u]['x'])
        train_data_num += user_train_data_num
        test_data_num += user_test_data_num
        # train_data_local_num_dict[client_index] = user_train_data_num

        # transform to batches

        train_data_x = train_data[u]['x']
        train_data_y = train_data[u]['y']
        test_data_x = test_data[u]['x']
        test_data_y = test_data[u]['y']

        # loop through mini-batches
        # batch_data = list()
        # for i in range(0, len(data_x), batch_size):
        #     batched_x = data_x[i:i + batch_size]
        #     batched_y = data_y[i:i + batch_size]
        client_train_x = torch.from_numpy(np.asarray(process_x(train_data_x)))
        client_train_y = torch.from_numpy(np.asarray(process_y(train_data_y)))
        client_test_x = torch.from_numpy(np.asarray(process_x(test_data_x)))
        client_test_y = torch.from_numpy(np.asarray(process_y(test_data_y)))
        # batch_data.append((batched_x, batched_y))
        all_train_data_x.append(client_train_x)
        all_train_data_y.append(client_train_y)
        all_test_data_x.append(client_test_x)
        all_test_data_y.append(client_test_y)


    all_train_data_x = torch.cat(all_train_data_x)
    all_train_data_y = torch.cat(all_train_data_y)
    all_test_data_x = torch.cat(all_test_data_x)
    all_test_data_y = torch.cat(all_test_data_y)


    # client_num = client_index
    output_dim = VOCAB_SIZE

    # train_x, train_y = utils.split(train_ds)
    # test_x, test_y = utils.split(test_ds)
    train_dataset = data.TensorDataset(all_train_data_x,
                                  all_train_data_y)
    test_dataset = data.TensorDataset(all_test_data_x,
                                 all_test_data_y)
    # train_dl = data.DataLoader(dataset=train_dataset,
    #                            batch_size=batch_size,
    #                            shuffle=True,
    #                            drop_last=False)
    # test_dl = data.DataLoader(dataset=test_dataset,
    #                           batch_size=batch_size,
    #                           shuffle=True,
    #                           drop_last=False)

    client_num = args.client_num_in_total
    client_number = client_num
    if args.mode == 'distributed':
        train_sampler = None
        shuffle = True
        if client_number > 1:
            train_sampler = data.distributed.DistributedSampler(
                train_dataset, num_replicas=client_number, rank=rank)
            train_sampler.set_epoch(0)
            shuffle = False

            # Note that test_sampler is for distributed testing to accelerate training
            test_sampler = data.distributed.DistributedSampler(
                test_dataset, num_replicas=client_number, rank=rank)
            train_sampler.set_epoch(0)


        train_data_global = data.DataLoader(train_dataset, batch_size=batch_size,
                                    shuffle=shuffle, num_workers=4, drop_last=True)
        test_data_global = data.DataLoader(test_dataset, batch_size=batch_size,
                                    shuffle=False, num_workers=4)

        train_sampler = train_sampler
        train_dl = data.DataLoader(train_dataset, batch_size=batch_size,
                                    shuffle=shuffle, num_workers=4, sampler=train_sampler, drop_last=True)
        test_dl = data.DataLoader(test_dataset, batch_size=batch_size,
                                    shuffle=False, num_workers=4)
        # classes = ('plane', 'car', 'bird', 'cat',
        #         'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


        train_data_num = len(train_dataset)
        test_data_num = len(test_dataset)

        data_local_num_dict = dict()
        train_data_local_dict = dict()
        test_data_local_dict = dict()

        for client_index in range(client_number):
            train_data_local_dict[client_index] = train_dl
            test_data_local_dict[client_index] = test_dl
            # Because the train_dataset has all samples, so here we divide it to get the length of local dataset.
            train_data_local_num_dict[client_index] = train_data_num // client_number
            logging.info("client_index = %d, local_sample_number = %d" % (client_index, train_data_num))
    elif args.mode == 'standalone':
        raise NotImplementedError
    else:
        raise NotImplementedError

    output_dim = VOCAB_SIZE

    return client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
           train_data_local_num_dict, train_data_local_dict, test_data_local_dict, output_dim
