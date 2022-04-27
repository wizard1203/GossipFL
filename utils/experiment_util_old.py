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




def combine_config(basic_config, update_config):
    new_config = deepcopy(basic_config)
    for k in list(update_config.keys()):
        new_config[k] = update_config[k]
    return new_config


def summary_name(mode, thing, layer=None, client_index=None):
    root = mode + '/' + thing
    if layer is not None:
        root += '/' + layer
    if client_index is not None:
        root += '/' + str(client_index)
    else:
        root += '/' + "global"
    return root


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
            # print("WARNING, The number of results found out is more than one.\
            #     There maybe duplicate experiments!!!!!!!")
            # out_df = out_df.sort_values(sort_value_name, ascending=ascending)
            # uid = out_df.iloc[0]["uid"]
            # exp_run = exp_book.get_run(uid)
            # detail_config = exp_run.config
            # print("The filtered detailed config is {}".format(
            #     detail_config
            #     ))
            # else:
            #     uid = out_df.iloc[0]["uid"]
            #     exp_run = exp_book.get_run(uid)
            return True
    except:
        return False



def find_one_uid(exp_book, config=None, help_params=None, filter_name=None, sort_value_name=None,
                sort=False, ascending=False):
    out_df = exp_book.all_df
    # filter according to config
    for key in config.keys():
        if key == "b_created_at":
            out_df = out_df.loc[out_df["created_at"] > config["b_created_at"]]
            continue
        elif key == "s_created_at":
            out_df = out_df.loc[out_df["created_at"] < config["s_created_at"]]
            continue
        else:
            pass

        if config[key] is None:
            out_df = out_df.loc[out_df[key].isnull()]
            continue
        else:
            pass

        if config[key] == 'no':
            out_df_no = out_df.loc[out_df[key] == config[key]]
            out_df_None = out_df.loc[out_df[key].isnull()]
            out_df = pd.concat([out_df_no, out_df_None])
            continue
        # if config[key] == 'no':
        #     test_out_df = out_df.loc[out_df[key] == config[key]]
        #     find_result = check_get_run(exp_book, test_out_df, sort_value_name, ascending)
        #     if find_result:
        #         # print("Find config: {} with key {} = no".format(config, key))
        #         out_df = test_out_df
        #         continue
        #     else:
        #         # print("Find config: {} with key {} = None".format(config, key))
        #         out_df = out_df.loc[out_df[key].isnull()]
        #         continue

        out_df = out_df.loc[out_df[key] == config[key]]



    # filter according to filter_name
    if len(out_df) > 0:
        print("len(out_df) is {} ".format(len(out_df)))
        test_df = out_df.loc[out_df[filter_name].notnull()]
        if len(test_df) > 0:
            out_df = test_df
        else:
            print("len(out_df) is {} there is no \"{}\" summary item".format(
                len(out_df), filter_name))

    # for key in config.keys():
    #     adf = out_df.loc[out_df[key] == config[key]]
    # # filter according to filter_name
    # adf = adf.loc[adf['Test/Acc'].notnull()]
    # sort runs according to sort_value_name
    if sort:
        try:
            if len(out_df) > 1:
                print("WARNING, The number of results found out is more than one.\
                    There maybe duplicate experiments!!!!!!!")
                out_df = out_df.sort_values(sort_value_name, ascending=ascending)
                uid = out_df.iloc[0]["uid"]
                exp_run = exp_book.get_run(uid)
                detail_config = exp_run.config
                print("The filtered detailed config is {}".format(
                    detail_config
                ))
            else:
                uid = out_df.iloc[0]["uid"]
                exp_run = exp_book.get_run(uid)
            try:
                print("config: {}, find uid successfully, is: {}, the {} is {}".format(
                    config, uid, filter_name, exp_run.summary[filter_name]))
            except:
                print("config: {}, find uid successfully, is: {}, but {} is not in summary".format(
                    config, uid, filter_name))
        except:
            print("ERROR!!!! config: {}, Not find uid".format(config))
            # print(out_df)
            # raise NotImplementedError
            uid = None
            exp_run = None
    else:
        uid = out_df["uid"]
    return uid, exp_run


class ExpRun(object):
    """        
    This class is designed for recording one experiment.
    For all experiments, You should use ExpBook.

    The definitions of name, state, config, created_at, system_metrics 
            summary, history and file are aligned with wandb:

    Attributes:
        id (str): unique identifier for the run (defaults to eight characters) (in one project)
        name (str): the name of the run
        state (str): one of: running, finished, crashed, aborted
        config (dict): a dict of hyperparameters associated with the run
        created_at (str): ISO timestamp when the run was started
        system_metrics (dict): the latest system metrics recorded for the run
        summary (dict): A mutable dict-like property that holds the current summary.
                    Calling update will persist any changes.
        history (pandas.core.frame.DataFrame): An pandas data frame recording all metrics of this run.
        file: Not used now.
        notes (str): Notes about the run
        #===================================================================
        uid (str): globel unique identifier for the run, in case you want to compare different 
                    experiment results. `project + '/' + id ` 
        url (str): identify the experiment result log path.
    """

    def __init__(self, entity, project, id, uid, config, run, 
                url=None, state=None, created_at=None, system_metrics=None,
                summary=None, file=None, generate_name=None):

        self.entity = entity 
        self.project = project
        self.id = id
        self.uid = uid
        assert self.uid == project + "/" + id
        self.config = config
        self.run = run

        self.url = url
        self.state = state 
        self.created_at = created_at
        self.system_metrics = system_metrics
        self.summary = summary

        # It will be better Not load running history here,
        self.history = None
        self.history_loaded = False
        self.file = file
        if generate_name is not None:
            self.generate_name = generate_name
        # TODO maybe the name is not useful, we just only need uid
        self.name = self.generate_name(config)


    def download_history(self, root_path):
        run_path = get_run_path(self.entity, self.project, self.created_at, self.id)
        self.history.to_csv(os.path.join(root_path, run_path))


    def download_file(self, file_name, folder_path, replace=True):
        self.run.file(file_name).download(folder_path, replace=replace)


    def download_wandb_files(self, folder_path, replace=True):
        self.run.file("requirements.txt").download(folder_path, replace=replace)
        self.run.file("config.yaml").download(folder_path, replace=replace)
        self.run.file("wandb-metadata.json").download(folder_path, replace=replace)
        self.run.file("wandb-summary.json").download(folder_path, replace=replace)



    def get_history(self):
        return self.history, self.history_loaded

    def refresh_history(self, download=True, unsample=False):
        """
        This function is used to 
        """
        try:
            if unsample:
                self.history = self.run.history()
            else:
                self.history = self.run.scan_history()
            self.history_loaded = True
        except Exception as e:
            print(e)
            traceback.print_exception(*sys.exc_info())
            self.history_loaded = False
            logging.info("ERROR!!!!  The {} exp does not have the history results!! \
                Try to loaded or check if it exist... ".format(self.config))
        return self.history, self.history_loaded



    def generate_name(self, config):
        """
        This class method is used for generate the name of one experiment record.
        Note that here the name is not unique, because we may want to change codes 
        between two experiments, or there may be only a part of arguments added into the name.
        Also, the name will be used as the name of the wandb run.
        """
        return generate_config_key(config)

    def get_history_metric(self, metric_name, filter_name, get_history_metric_func=None):
        """ It will be better to use pandas' own filter"""
        if get_history_metric_func is None:
            return list(self.history.loc[self.history[filter_name].notnull()][metric_name])
        else:
            return get_history_metric_func(self.history)


    def get_summary_metric(self, metric_name, filter_name, get_summary_metric_func=None):
        """ It will be better to use pandas' own filter"""
        if get_summary_metric_func is None:
            return list(self.summary.loc[self.summary[filter_name].notnull()][metric_name])
        else:
            return get_summary_metric_func(self.summary)


class ExpBook(object):
    """
    This class has all experiment runs. 
    You can filter the `all_df` to get the uid. Then use uid to get runs.
    The definitions of name, state, config, created_at, system_metrics 
            summary and file are aligned with wandb:

    Attributes:
        all_df (pandas.core.frame.DataFrame): all informations of runs
        runs (dict): all runs, {uid (str): run (ExpRun)}
                        uid should be unique.
    """

    def __init__(self, all_df, runs_dict):
        self.all_df = all_df
        self.runs = {}
        for key in runs_dict.keys():
            # self.runs[key] = ExpRun(
            #     runs_dict[key].project, runs_dict[key].id, key,
            #     runs_dict[key].config, runs_dict[key].url, runs_dict[key].state,
            #     runs_dict[key].created_at, runs_dict[key].system_metrics,
            #     runs_dict[key].summary, runs_dict[key].history(), runs_dict[key].file
            # )
            self.runs[key] = ExpRun(
                runs_dict[key].entity,
                runs_dict[key].project, runs_dict[key].id, key, runs_dict[key].config,
                runs_dict[key], runs_dict[key].url, runs_dict[key].state,
                runs_dict[key].created_at, runs_dict[key].system_metrics,
                runs_dict[key].summary, runs_dict[key].file
            )

    def get_run(self, uid):
        return self.runs[uid]

    def get_runs(self, uid_list=None, config=None, filter_name=None, sort_value_name=None,
                sort=False, ascending=False, choosing_num=1):
        """
        You can use this method to get runs that you want.
        This method is very simple. You can directly use `all_df` to get runs.
        """
        out_runs = {}
        if uid_list is not None:
            for uid in uid_list:
                out_runs[uid] = self.get_run(uid)
        else:
            out_df = self.all_df

            # filter according to config
            for key in config.keys():
                if config[key] is None:
                    out_df = out_df.loc[out_df[key].isnull()]
                else:
                    out_df = out_df.loc[out_df[key == config[key]]]


            # filter according to filter_name
            out_df = out_df.loc[out_df[filter_name].notnull()]

            # sort runs according to sort_value_name
            if sort:
                out_df = out_df.sort_values(sort_value_name, ascending=ascending).iloc[0:choosing_num]

            # get runs
            for item in out_df.iterrows():
                out_runs[item["uid"]] = self.get_run(item["uid"])
        return out_runs




    # TODO  I do not have thounght out how to write a good filter function.
    # Maybe directly using pd_frame to filter is good enough.
    def filter(self):
        pass



class ExpPlot(object):
    """
    This class aim to contain a set of runs which will be shown in one figure.
    We want to choose some runs with specifit configs to show and compare.
    Here we will define some tools to help to show the figures.

    Attributes:
        config_groups (dict):
            {config_key (str): {
                "config": config (str),
                "uid": uid (str)
                "help_params": a dict of help_params
                }, ...
            }
        # TODO Maybe we do not need put runs here. We can just find runs from ExpBook
        # runs (dict): all runs, {uid (str): run (ExpRun)}
        #             uid should be unique.
        config_alias (dict): alias of config_keys, for convenience of customizing config key.
    """
    def __init__(self, exp_book: ExpBook, filter_name, sort_value_name, sort, ascending):
        self.config_groups = {}
        self.runs_groups = {}
        # self.runs = {}
        self.config_alias = {}
        self.exp_book = exp_book
        self.filter_name = filter_name
        self.sort_value_name = sort_value_name
        self.sort = sort
        self.ascending = ascending


    def add_config(self, config, alias=None, uid=None, help_params=None, find_uid_func=None):
        """
        By this method, you can register `find_one_uid`, then we can find
        runs by `find_one_uid()`.
        Here the `config` does not need to have all args in experiment, in case you want to 
        sort the filtered results and choose the highest or lowest one.
        Every config shoule be unique in one ExpPlot.

        Arguments:
            config (dict): all args in config shoule also be a subset of ExpRun.config.
            uid (str):  globel unique identifier for the run, in case you want to compare different 
                        experiment results. `project + '/' + id `.
            help_params (dict): The help_params is used to plot more figures with parameters
                                that are not included in ExpRun.config, in case you want to 
                                make some more powerful expressions.
        """
        config_key = generate_config_key(config)
        if uid is not None:
            exp_run = self.exp_book.get_run(uid)
        elif find_uid_func is None:
            uid, exp_run = find_one_uid(self.exp_book, config, help_params, self.filter_name, self.sort_value_name,
                            self.sort, self.ascending)
        elif find_uid_func is not None:
            uid, exp_run = find_uid_func(self.exp_book, config, help_params, self.filter_name, self.sort_value_name,
                            self.sort, self.ascending)

        self.config_groups[config_key] = {
            "alias": alias,
            "config": config,
            "uid": uid,
            "help_params": help_params,
            "exp_run": exp_run,
            "loaded": False
        }
        if alias is not None:
            self.config_alias[alias] = config_key
        return config_key, uid, exp_run

    # TODO future feature
    # def arguments_parse():

    def get_group(self, config_key=None, config=None, alias=None):
        """
        Get group
        """
        if config_key is not None:
            group = self.config_groups[config_key]
        if alias is not None:
            group = self.config_groups[self.config_alias[alias]]
        if config is not None:
            group = self.config_groups[generate_config_key(config)]
        return group

    def get_config(self, config_key=None, alias=None):
        assert (config_key is not None) or (alias is not None)
        if config_key is None:
            config_key = self.config_alias[alias] 
        return self.config_groups[config_key]["config"]


    def get_uid(self, config_key=None, config=None, alias=None):
        # assert (config_key is not None) or (config is not None) or (alias is not None)
        group = self.get_group(config_key, config, alias)
        return group["uid"]


    def get_help_params(self, config_key=None, config=None, alias=None):
        # assert (config_key is not None) or (config is not None) or (alias is not None)
        group = self.get_group(config_key, config, alias)
        return group["help_params"]


    def set_uid(self, uid, config_key=None, config=None, alias=None):
        # assert (config_key is not None) or (config is not None) or (alias is not None)
        group = self.get_group(config_key, config, alias)
        group["uid"] = uid



    def set_help_params(self, help_params, config_key=None, config=None, alias=None):
        # assert (config_key is not None) or (config is not None) or (alias is not None)
        # if config_key is not None:
        #     self.config_groups[config_key]["help_params"] = help_params
        # if alias is not None:
        #     self.config_groups[self.config_alias[alias]]["help_params"] = help_params
        # if config is not None:
        #     self.config_groups[generate_config_key(config)]["help_params"] = help_params
        group = self.get_group(config_key, config, alias)
        group["help_params"] = help_params


    def refresh_wandb_history(self):
        history_status = {}
        find_run = True
        for config_key, group in self.config_groups.items():
            if group["loaded"] is False and group["exp_run"] is not None:
                try:
                    group["exp_run"].refresh_history()
                    group["loaded"] = True
                    history_status[config_key] = True
                except Exception as e:
                    print(e)
                    traceback.print_exception(*sys.exc_info())
                    find_run = False
                    history_status[config_key] = False
            else:
                history_status[config_key] = True
        return find_run, history_status


    def get_wandb_history(self, config_key=None, config=None, alias=None):
        """
        Get run history
        """
        group = self.get_group(config_key, config, alias)
        exp_run = group["exp_run"]

        if group["loaded"]:
            find_history = True
            history = exp_run.history
        else:
            if exp_run is not None:
                try:
                    exp_run.refresh_history()
                    history = exp_run.history
                    group["loaded"] = True
                    find_history = True
                except Exception as e:
                    print(e)
                    traceback.print_exception(*sys.exc_info())
                    find_history = False
                    logging.info("ERROR!!!!  The {} exp does not have the history results!! \
                        Try to loaded or check if it exist... ".format(group["config"]))

        return find_history, group, history


    def get_file(self, config_key=None, config=None, alias=None,
                file_name=None, file_path=None, local_backup=False):
        """
        Get run file
        """




    # def get_wandb_history_x_and_y(self, config_key=None, config=None, alias=None,
    #         get_x_func=None, get_y_func=None):

    #     find_history, group, history = self.get_wandb_history(config_key, config, alias)

    #     exp_run = group["exp_run"]
    #     config = group["config"]
    #     help_params = group["help_params"]
    #     alias = group["alias"]

    #     if find_history:
    #         try:
    #             x = get_x_func(history, alias, config, help_params)
    #             y = get_y_func(history, alias, config, help_params)
    #             max_x = max(x)
    #             max_y = max(y)
    #         except Exception as e:
    #             print(e)
    #             traceback.print_exception(*sys.exc_info())
    #             x = np.array([0])
    #             y = np.array([0])
    #             max_x = None
    #             max_y = None
    #     else:
    #         x = np.array([0])
    #         y = np.array([0])
    #         max_x = None
    #         max_y = None
    #     return x, max_x, y, max_y


    def get_run(self, config_key=None, config=None, alias=None):
        """
        Get run from ExpBook
        """
        # if config_key is not None:
        #     # uid = self.config_groups[config_key]["uid"]
        #     exp_run = self.config_groups[config_key]["exp_run"]
        # if alias is not None:
        #     # uid = self.config_groups[self.config_alias[alias]]["uid"]
        #     exp_run = self.config_groups[self.config_alias[alias]]["exp_run"]
        # if config is not None:
        #     # uid = self.config_groups[generate_config_key(config)]["uid"]
        #     exp_run = self.config_groups[generate_config_key(config)]["exp_run"]
        # logging.info("No input")
        # return self.exp_book.get_run(uid)

        group = self.get_group(config_key, config, alias)
        return group["exp_run"]





