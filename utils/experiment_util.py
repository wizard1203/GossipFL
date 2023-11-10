import logging
from copy import deepcopy
import pandas as pd

def combine_config(basic_config, update_config):
    new_config = deepcopy(basic_config)
    for k in list(update_config.keys()):
        new_config[k] = update_config[k]
    return new_config



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

    def __init__(self, project, id, uid, config, run, url=None, state=None, created_at=None, system_metrics=None,
                 summary=None, file=None, generate_name=None):

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
        self.file = file
        if generate_name is not None:
            self.generate_name = generate_name
        # TODO maybe the name is not useful, we just only need uid
        self.name = self.generate_name(config)



    def refresh_history(self):
        """
        This function is used to 
        """
        self.history = self.run.history()


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
            exp_run = None
        elif find_uid_func is None:
            uid, exp_run = find_one_uid(self.exp_book, config, help_params, self.filter_name, self.sort_value_name,
                               self.sort, self.ascending)
        elif (find_uid_func is not None):
            uid, exp_run = find_uid_func(self.exp_book, config, help_params, self.filter_name, self.sort_value_name,
                               self.sort, self.ascending)

        self.config_groups[config_key] = {
            "config": config,
            "uid": uid,
            "help_params": help_params
        }
        if alias is not None:
            self.config_alias[alias] = config_key
        return config_key, uid, exp_run

    # TODO future feature
    # def arguments_parse():


    def get_config(self, config_key=None, alias=None):
        assert (config_key is not None) or (alias is not None)
        if config_key is None:
            config_key = self.config_alias[alias] 
        return self.config_groups[config_key]["config"]


    def get_uid(self, config_key=None, config=None, alias=None):
        # assert (config_key is not None) or (config is not None) or (alias is not None)
        if config_key is not None:
            return self.config_groups[config_key]["uid"]
        if alias is not None:
            return self.config_groups[self.config_alias[alias]]["uid"]
        if config is not None:
            return self.config_groups[generate_config_key(config)]["uid"]
        logging.info("No input")


    def get_help_params(self, config_key=None, config=None, alias=None):
        # assert (config_key is not None) or (config is not None) or (alias is not None)
        if config_key is not None:
            return self.config_groups[config_key]["help_params"]
        if alias is not None:
            return self.config_groups[self.config_alias[alias]]["help_params"]
        if config is not None:
            return self.config_groups[generate_config_key(config)]["help_params"]
        logging.info("No input")


    def set_uid(self, uid, config_key=None, config=None, alias=None):
        # assert (config_key is not None) or (config is not None) or (alias is not None)
        if config_key is not None:
            self.config_groups[config_key]["uid"] = uid
        if alias is not None:
            self.config_groups[self.config_alias[alias]]["uid"] = uid
        if config is not None:
            self.config_groups[generate_config_key(config)]["uid"] = uid
        logging.info("No input")


    def set_help_params(self, help_params, config_key=None, config=None, alias=None):
        # assert (config_key is not None) or (config is not None) or (alias is not None)
        if config_key is not None:
            self.config_groups[config_key]["help_params"] = help_params
        if alias is not None:
            self.config_groups[self.config_alias[alias]]["help_params"] = help_params
        if config is not None:
            self.config_groups[generate_config_key(config)]["help_params"] = help_params
        logging.info("No input")


    def get_run(self, config_key=None, config=None, alias=None):
        """
        Get run from ExpBook
        """
        if config_key is not None:
            uid = self.config_groups[config_key]["uid"]
        if alias is not None:
            uid = self.config_groups[self.config_alias[alias]]["uid"]
        if config is not None:
            uid = self.config_groups[generate_config_key(config)]["uid"]
        logging.info("No input")
        return self.exp_book.get_run(uid)






