import os
import logging



from .loader import Data_Loader
from .loader_shakespeare import Shakespeare_Data_Loader
from .generative_loader import Generative_Data_Loader

from .loader import NORMAL_DATASET_LIST
from .loader_shakespeare import SHAKESPEARE_DATASET_LIST
from .generative_loader import GENERATIVE_DATASET_LIST


def get_new_datadir(args, datadir, dataset):
    # if "style_GAN_init" in dataset or "Gaussian" in dataset or "decoder" in dataset:
    if dataset in GENERATIVE_DATASET_LIST:
        return os.path.join(args.generative_dataset_root_path, dataset)
    else:
        return datadir



def load_data(load_as, args=None, process_id=0, mode="centralized", task="centralized", data_efficient_load=True,
                dirichlet_balance=False, dirichlet_min_p=None,
                dataset="", datadir="./", partition_method="iid", partition_alpha=0.5, client_number=1, batch_size=128, num_workers=4,
                data_sampler=None,
                resize=32, augmentation="default"):

    datadir = get_new_datadir(args, datadir, dataset)
    other_params = {}

    if task == "centralized":
        assert mode == "centralized"
        assert task == "centralized"
        if load_as == "training":
            if dataset in NORMAL_DATASET_LIST:
                data_loader = Data_Loader(args=args, process_id=process_id, mode=mode, task=task,
                    dataset=dataset, datadir=datadir, batch_size=batch_size, num_workers=num_workers,
                    data_sampler=data_sampler,
                    resize=resize, augmentation=augmentation, other_params=other_params)
                # train_dl, test_dl, train_data_num, test_data_num, class_num, other_params = data_loader.load_centralized_data()
                train_dl, test_dl, train_data_num, test_data_num, class_num, other_params = data_loader.load_data()
                train_ds = data_loader.train_ds
                test_ds = data_loader.test_ds
            elif dataset in SHAKESPEARE_DATASET_LIST:
                data_loader = Shakespeare_Data_Loader(args=args, process_id=process_id, mode=mode, task=task,
                    dataset=dataset, datadir=datadir, batch_size=batch_size, num_workers=num_workers,
                    data_sampler=data_sampler,
                    resize=resize, augmentation=augmentation, other_params=other_params)
                # train_dl, test_dl, train_data_num, test_data_num, class_num, other_params = data_loader.load_centralized_data()
                train_dl, test_dl, train_data_num, test_data_num, class_num, other_params = data_loader.load_data()
                train_ds = data_loader.train_ds
                test_ds = data_loader.test_ds

            elif dataset in GENERATIVE_DATASET_LIST:
                data_loader = Generative_Data_Loader(args=args, process_id=process_id, mode=mode, task=task,
                    dataset=dataset, datadir=datadir, batch_size=batch_size, num_workers=num_workers,
                    data_sampler=data_sampler,
                    resize=resize, augmentation=augmentation, other_params=other_params)
                # train_dl, test_dl, train_data_num, test_data_num, class_num, other_params = data_loader.load_centralized_data()
                train_dl, test_dl, train_data_num, test_data_num, class_num, other_params = data_loader.load_data()
                train_ds = data_loader.train_ds
                test_ds = data_loader.test_ds
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        other_params["train_ds"] = train_ds
        other_params["test_ds"] = test_ds
        return train_dl, test_dl, train_data_num, test_data_num, class_num, other_params
    else:
        if load_as == "training":
            if dataset in NORMAL_DATASET_LIST:
                data_loader = Data_Loader(args, process_id, mode, task, data_efficient_load, dirichlet_balance, dirichlet_min_p,
                    dataset, datadir, partition_method, partition_alpha, client_number, batch_size, num_workers,
                    data_sampler,
                    resize=resize, augmentation=augmentation, other_params=other_params)
                train_data_num, test_data_num, train_data_global, test_data_global, \
                    data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, other_params \
                        = data_loader.load_data()
            elif dataset in SHAKESPEARE_DATASET_LIST:
                data_loader = Shakespeare_Data_Loader(args, process_id, mode, task, data_efficient_load, dirichlet_balance, dirichlet_min_p,
                    dataset, datadir, partition_method, partition_alpha, client_number, batch_size, num_workers,
                    data_sampler,
                    resize=resize, augmentation=augmentation, other_params=other_params)
                train_data_num, test_data_num, train_data_global, test_data_global, \
                    data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, other_params \
                        = data_loader.load_data()
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return train_data_num, test_data_num, train_data_global, test_data_global, \
                data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, other_params






def load_multiple_centralized_dataset(load_as, args, process_id, mode, task,
                        dataset_list, datadir_list, batch_size, num_workers,
                        data_sampler=None,
                        resize=32, augmentation="default"): 
    train_dl_dict = {}
    test_dl_dict = {}
    train_ds_dict = {}
    test_ds_dict = {}
    class_num_dict = {}
    train_data_num_dict = {}
    test_data_num_dict = {}

    for i, dataset in enumerate(dataset_list):
        # kwargs["data_dir"] = datadir_list[i]
        datadir = datadir_list[i]
        # train_dl, test_dl, train_data_num, test_data_num, class_num, other_params \
        #     = load_centralized_data(load_as, args, process_id, mode, task,
        #                 dataset, datadir, batch_size, num_workers,
        #                 data_sampler=None,
        #                 resize=resize, augmentation=augmentation)
        train_dl, test_dl, train_data_num, test_data_num, class_num, other_params \
            = load_data(load_as=load_as, args=args, process_id=process_id,
                        mode="centralized", task="centralized",
                        dataset=dataset, datadir=datadir, batch_size=args.batch_size, num_workers=args.data_load_num_workers,
                        data_sampler=None,
                        resize=resize, augmentation=augmentation)

        train_dl_dict[dataset] = train_dl
        test_dl_dict[dataset] = test_dl
        train_ds_dict[dataset] = other_params["train_ds"]
        test_ds_dict[dataset] = other_params["test_ds"]
        class_num_dict[dataset] = class_num
        train_data_num_dict[dataset] = train_data_num
        test_data_num_dict[dataset] = test_data_num

    return train_dl_dict, test_dl_dict, train_ds_dict, test_ds_dict, \
        class_num_dict, train_data_num_dict, test_data_num_dict












