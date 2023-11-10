import os
import logging

import torch

def setup_checkpoint_config(args):
    assert args.checkpoint_save is True
    save_checkpoints_config = {}
    save_checkpoints_config["model_state_dict"] = True if args.checkpoint_save_model else False
    save_checkpoints_config["optimizer_state_dict"] = True if args.checkpoint_save_optim else False
    save_checkpoints_config["train_metric_info"] = True if args.checkpoint_save_train_metric else False
    save_checkpoints_config["test_metric_info"] = True if args.checkpoint_save_test_metric else False
    save_checkpoints_config["checkpoint_root_path"] = args.checkpoint_root_path
    save_checkpoints_config["checkpoint_epoch_list"] = args.checkpoint_epoch_list
    save_checkpoints_config["checkpoint_file_name_save_list"] = args.checkpoint_file_name_save_list
    save_checkpoints_config["checkpoint_file_name_prefix"] = setup_checkpoint_file_name_prefix(args)
    return save_checkpoints_config


def setup_checkpoint_file_name_prefix(args):
    checkpoint_file_name_prefix = ""
    for i, name in enumerate(args.checkpoint_file_name_save_list):
        checkpoint_file_name_prefix += str(getattr(args, name))
        if i != len(args.checkpoint_file_name_save_list) - 1:
            checkpoint_file_name_prefix += "-"
    return checkpoint_file_name_prefix

def setup_save_checkpoint_path(save_checkpoints_config, extra_name=None, epoch="init"):
    if extra_name is not None:
        checkpoint_path = save_checkpoints_config["checkpoint_root_path"] \
            + "checkpoint-" + extra_name + save_checkpoints_config["checkpoint_file_name_prefix"] \
            + "-epoch-"+str(epoch) + ".pth"
    else:
        checkpoint_path = save_checkpoints_config["checkpoint_root_path"] \
            + "checkpoint-" + extra_name + save_checkpoints_config["checkpoint_file_name_prefix"] \
            + "-epoch-"+str(epoch) + ".pth"

    return checkpoint_path 

def save_checkpoint(args, save_checkpoints_config, extra_name=None, epoch="init",
                    model_state_dict=None, optimizer_state_dict=None,
                    train_metric_info=None, test_metric_info=None):
    if save_checkpoints_config is None:
        logging.info("WARNING: Not save checkpoints......")
        return
    if epoch in save_checkpoints_config["checkpoint_epoch_list"]:
        checkpoint_path = setup_save_checkpoint_path(save_checkpoints_config, extra_name, epoch)
        if not os.path.exists(save_checkpoints_config["checkpoint_root_path"]):
            os.makedirs(save_checkpoints_config["checkpoint_root_path"])
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict if save_checkpoints_config["model_state_dict"] else None,
            'optimizer_state_dict': optimizer_state_dict if save_checkpoints_config["optimizer_state_dict"] else None,
            'train_metric_info': train_metric_info if save_checkpoints_config["train_metric_info"] else None,
            'test_metric_info': test_metric_info if save_checkpoints_config["test_metric_info"] else None,
            }, checkpoint_path)
        logging.info("WARNING: Saving checkpoints {} at epoch {}......".format(
            checkpoint_path, epoch))
    else:
        logging.info("WARNING: Not save checkpoints......")









