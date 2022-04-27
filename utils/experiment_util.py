import logging
import os
import sys
import traceback
from copy import deepcopy

import pandas as pd
import numpy as np

from .wandb_util import (
    get_project_path,
    get_run_folder_name,
    get_run_path,
    time_to_number_str,
    number_str_to_time
)

import wandb

def get_alias(config):
    return "-".join([str(key)+"="+str(config[key]) for key in config.keys()])


def get_alias_from_list(*args):
    return "-".join([str(arg) for arg in args])


# def add_alias(alias_list, alias_run_map, alias, config, help_params={},
#     uid=None, find_uid_func=None, find_uid_kargs=None):
#     alias_run_map.append(alias)
#     alias_run_map[alias] = {
#         "config": config,
#         "help_params": help_params,
#         "uid": uid,
#         "find_uid_func": find_uid_func,
#         "find_uid_kargs": find_uid_kargs
#     }


def generate_alias_list(
    alias_run_map, exp_book,
    basic_config={}, fixed_config={}, varying_config_list=[],
    fixed_help_params={}, varing_help_params_list=[],
    uid_list=[], find_uid_func=None, find_uid_kargs=None):
    """
        use alias_run_map to record,
        All config parameters should be same as config on wandb.
    """

    config_list = []
    help_params_list = []
    alias_list = []

    new_basic_config = combine_config(basic_config, fixed_config)

    for varying_config in varying_config_list:
        config_list.append(combine_config(new_basic_config, varying_config))

    basic_help_params = {}
    basic_help_params = combine_config(basic_help_params, fixed_help_params)

    if len(varing_help_params_list) > 0:
        for varing_help_params in varing_help_params_list:
            help_params_list.append(combine_config(basic_help_params, varing_help_params))

    for i, config in enumerate(config_list):
        if len(varing_help_params_list) > 0:
            help_params = help_params_list[i]
        else:
            help_params = basic_help_params

        if len(uid_list) > 0:
            uid = uid_list[i]
        else:
            uid = None

        alias = get_alias(config)
        alias_list.append(alias)
        alias_run_map[alias] = {
            "config": config,
            "help_params": help_params,
            "uid": uid,
            "find_uid_func": find_uid_func,
            "find_uid_kargs": find_uid_kargs
        }
        exp_book.add_config(
            config=config, alias=alias,
            help_params=help_params, uid=uid,
            find_uid_func=find_uid_func)
    return alias_list



def modify_dict(dict, key_func=None, value_func=None):
    new_dict = {}
    for key, value in dict.items():
        if key_func is not None:
            new_dict[key_func(key)] = value_func(value)
        else:
            new_dict[key] = value_func(value)
    return new_dict


def modify_list(list, value_func=None):
    new_list = []
    for value in list:
        new_list.append(value_func(value))
    return new_list


def combine_config(basic_config, update_config):
    new_config = deepcopy(basic_config)
    for k in list(update_config.keys()):
        new_config[k] = update_config[k]
    return new_config

def extend_dicts_from_list(old_dict_list, name, value_list):
    new_dict_list = []

    for old_dict in old_dict_list:
        if value_list is None or len(value_list) == 0:
            new_dict = deepcopy(old_dict)
            new_dict[name] = None
            new_dict_list.append(new_dict)
        else:
            for value in value_list:
                new_dict = deepcopy(old_dict)
                new_dict[name] = value
                new_dict_list.append(new_dict)
    return new_dict_list


def build_dicts(dict_of_list_attributes, basic_dict_list):
    new_dict_list = basic_dict_list
    for name, value_list in dict_of_list_attributes.items():
        new_dict_list = extend_dicts_from_list(
            old_dict_list=new_dict_list,
            name=name,
            value_list=value_list
        )
    return new_dict_list



def postfix_process(
    client_index=None,
    server_index=None,
    if_global=False
):
    postfix = ""
    if client_index is not None and \
        (server_index is None and if_global is False):
        postfix += '/client' + str(client_index)
    elif server_index is not None and \
        (client_index is None and if_global is False):
        postfix += '/server' + str(server_index)
    elif if_global and \
        (server_index is None and client_index is None):
        postfix += '/' + "global"
    elif if_global is False and \
        (server_index is None and client_index is None):
        pass
    else:
        raise RuntimeError

    return postfix


# def get_summary_name(
#     line_mode=None, thing=None,
#     LP=None,
#     layer=None, 
#     client_index=None, server_index=None,
#     if_global=False,
#     line_params=None
# ):
#     if line_params is None:
#         root = line_mode + '/' + thing

#         if LP is not None:
#             root += '/LP' + str(LP) 

#         if layer is not None:
#             root += '/' + layer

#         root += postfix_process(
#             client_index, server_index, if_global
#         )

#     else:
#         root = line_params["line_mode"] + '/' + line_params["thing"]
#         if line_params["LP"] is not None:
#             root += '/LP' + str(line_params["LP"]) 

#         if line_params["layer"] is not None:
#             root += '/' + line_params["layer"]

#         root += postfix_process(
#             line_params["client_index"],
#             line_params["server_index"], 
#             line_params["if_global"]
#         )

#     return root


def return_name_in_dict(name, dict, default):
    if name in dict:
        return dict[name]
    else:
        return default


def get_summary_name(
    **summary_name_dict
):
    root = summary_name_dict["line_mode"] + '/' + summary_name_dict["thing"]

    if "LP" in summary_name_dict and summary_name_dict["LP"] is not None:
        root += '/LP' + str(summary_name_dict["LP"]) 

    if "layer" in summary_name_dict and summary_name_dict["layer"] is not None:
        root += '/' + summary_name_dict["layer"]

    root += postfix_process(
        client_index=return_name_in_dict("client_index", summary_name_dict, None),
        server_index=return_name_in_dict("server_index", summary_name_dict, None),
        if_global=return_name_in_dict("if_global", summary_name_dict, False),
    )

    return root



def strip_summary_name(total_summary_name):
    stripped_summary_name_list = total_summary_name.split("/")
    return stripped_summary_name_list[1]


def get_metric_params(line_mode, thing, layers=None, LP_list=None,
        client_list=None, server_list=None, if_global=False
    ):
    basic_dict = {
        "line_mode": line_mode,
        "thing": thing
    }

    new_dict_list = [basic_dict]

    if layers is not None:
        new_dict_list = build_dicts(
            dict_of_list_attributes={"layer": layers},
            basic_dict_list=new_dict_list
        )

    if LP_list is not None:
        new_dict_list = build_dicts(
            dict_of_list_attributes={"LP": LP_list},
            basic_dict_list=new_dict_list
        )

    output_dict_list = []
    if client_list is not None and len(client_list) > 0:
        output_dict_list += build_dicts(
            dict_of_list_attributes={"client_index": client_list},
            basic_dict_list=new_dict_list
        )


    if server_list is not None and len(server_list) > 0:
        output_dict_list += build_dicts(
            dict_of_list_attributes={"server_index": server_list},
            basic_dict_list=new_dict_list
        )

    if if_global:
        output_dict_list += build_dicts(
            dict_of_list_attributes={"if_global": [if_global]},
            basic_dict_list=new_dict_list
        )

    if client_list is None and server_list is None and not if_global:
        output_dict_list = new_dict_list

    return output_dict_list



def get_metric_things(line_mode, thing, layers=None, LP_list=None,
        client_list=None, server_list=None, if_global=False
    ):

    output_dict_list = get_metric_params(line_mode, thing,
        layers=layers, LP_list=LP_list,
        client_list=client_list, server_list=server_list,
        if_global=if_global)

    things_list = []
    for output_dict in output_dict_list:
        things_list.append(get_summary_name(**output_dict))

    return things_list



# def get_metric_things(line_mode, thing, layers=None, LP_list=None,
#         client_list=None, if_global=False
#     ):
#     things_list = []
#     if layers is not None:
#         for layer in layers:
#             if client_list is not None:
#                 for client in client_list:
#                     things_list.append(get_summary_name(
#                         line_mode=line_mode, thing=thing, LP=LP, layer=layer, client_index=client))

#             things_list.append(get_summary_name(
#                 line_mode=line_mode, thing=thing, LP=LP, layer=layer, if_global=if_global))

#     else:
#         if client_list is not None:
#             for client in client_list:
#                 things_list.append(get_summary_name(
#                     line_mode=line_mode, thing=thing, LP=LP, client_index=client))

#         things_list.append(get_summary_name(
#             line_mode=line_mode, thing=thing, LP=LP, if_global=if_global))

#     return things_list



def get_same_alias_metric_things(alias_list, 
    line_mode, thing, layers=None, client_list=None, if_global=False
):
    things_list = get_metric_things(
        line_mode, thing, layers=None, client_list=None, if_global=if_global)

    logging.info(f"things_list: {things_list}")

    alias_metric_things_dict = {}
    for alias in alias_list:
        alias_metric_things_dict[alias] = things_list
    return alias_metric_things_dict


def generate_config_key(config):
    """
    This method is used to generate the name of one experiment record.
    Note that here the name is not unique, because we may want to change codes 
    between two experiments, or there may be only a part of arguments added into the name.
    Also, the name will be used as the name of the wandb run.
    """
    return "-".join([str(key)+"="+str(config[key]) for key in config.keys()])


def get_legend_name(prefix_and_names: list, map_seg='', seg='-'):
    """
    Each item in prefix_and_names shoule be (prefix, name)
    """
    legend_name = ""
    for i, item in enumerate(prefix_and_names):
        if i == prefix_and_names:
            legend_name += str(item[0]) + map_seg + str(item[1]) + seg
        else:
            legend_name += str(item[0]) + map_seg + str(item[1])


def check_get_run(exp_book, out_df, sort_value_name, ascending):
    try:
        if len(out_df) > 0:
            # logging.info("WARNING, The number of results found out is more than one.\
            #     There maybe duplicate experiments!!!!!!!")
            # out_df = out_df.sort_values(sort_value_name, ascending=ascending)
            # uid = out_df.iloc[0]["uid"]
            # exp_run = exp_book.get_run(uid)
            # detail_config = exp_run.config
            # logging.info("The filtered detailed config is {}".format(
            #     detail_config
            #     ))
            # else:
            #     uid = out_df.iloc[0]["uid"]
            #     exp_run = exp_book.get_run(uid)
            return True
    except:
        return False

















