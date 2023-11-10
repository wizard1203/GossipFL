import logging
from copy import deepcopy


class MaxMeter(object):
    """
    Keeps track of the max of all the values that are 'add'ed
    """

    def __init__(self):
        self.max = None

    def update(self, value):
        """
        Add a value to the accumulator.
        :return: `true` if the provided value became the new max
        """
        if self.max is None or value > self.max:
            self.max = deepcopy(value)
            return True
        else:
            return False

    def value(self):
        """Access the current running average"""
        return self.max


class MinMeter(object):
    """
    Keeps track of the max of all the values that are 'add'ed
    """

    def __init__(self):
        self.min = None

    def update(self, value):
        """
        Add a value to the accumulator.
        :return: `true` if the provided value became the new max
        """
        if self.min is None or value < self.min:
            self.min = deepcopy(value)
            return True
        else:
            return False

    def value(self):
        """Access the current running average"""
        return self.min


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.max = -float("inf")
        self.min = float("inf")
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = val if val > self.max else self.max
        self.min = val if val < self.min else self.min


class RuntimeTracker(object):
    """Tracking the runtime stat for local training."""

    # def __init__(self, metrics_to_track=["top1"], on_cuda=True):
    def __init__(self, things_to_track=["loss"], on_cuda=True, id=None):
        self.things_to_track = things_to_track
        self.on_cuda = on_cuda
        self.n_samples = 0
        self.time_stamp = 0
        self.id = id
        self.stat = None
        self.reset()

    def reset(self):
        self.stat = dict((name, AverageMeter()) for name in self.things_to_track)
        self.n_samples = 0

    # def evaluate_global_metric(self, metric):
    #     return global_average(
    #         self.stat[metric].sum, self.stat[metric].count, on_cuda=self.on_cuda
    #     ).item()

    # def evaluate_global_metrics(self):
    #     return [self.evaluate_global_metric(metric) for metric in self.metrics_to_track]

    def get_metrics_performance(self):
        return [self.stat[thing].avg for thing in self.things_to_track]

    def update_metrics(self, metric_stat, n_samples):
        if n_samples == 0 or n_samples < 0:
            logging.info("WARNING: update_metrics received n_samples = 0 or < 0!!!!!!")
            return
        self.n_samples += n_samples
        for thing in self.things_to_track:
            self.stat[thing].update(metric_stat[thing], n_samples)

    def update_time_stamp(self, time_stamp):
        self.time_stamp = time_stamp

    def get_metric_info(self):
        metric_info = dict((name, val.avg) for name, val in self.stat.items())
        metric_info['n_samples'] = self.n_samples
        metric_info['time_stamp'] = self.time_stamp
        return metric_info

    def __call__(self, metrics=None, **kargs):
        if metrics is None or metrics.get_metric_info_func is None:
            metric_info = dict((name, val.avg) for name, val in self.stat.items())
            metric_info['n_samples'] = self.n_samples
            metric_info['time_stamp'] = self.time_stamp
        else:
            metric_info = metrics.get_metric_info_func(self.stat, **kargs)
            metric_info['n_samples'] = self.n_samples
            metric_info['time_stamp'] = self.time_stamp

        return metric_info


class BestPerf(object):
    def __init__(self, best_perf=None, larger_is_better=True):
        self.best_perf = best_perf
        self.cur_perf = None
        self.best_perf_locs = []
        self.larger_is_better = larger_is_better

        # define meter
        self._define_meter()

    def _define_meter(self):
        self.meter = MaxMeter() if self.larger_is_better else MinMeter()

    def update(self, perf, perf_location):
        self.is_best = self.meter.update(perf)
        self.cur_perf = perf

        if self.is_best:
            self.best_perf = perf
            self.best_perf_locs += [perf_location]

    def get_best_perf_loc(self):
        return self.best_perf_locs[-1] if len(self.best_perf_locs) != 0 else None



def get_metric_info(train_tracker, test_tracker, time_stamp, if_reset, metrics=None):
    train_tracker.update_time_stamp(time_stamp=time_stamp)
    train_metric_info = train_tracker(metrics)
    test_tracker.update_time_stamp(time_stamp=time_stamp)
    test_metric_info = test_tracker(metrics)

    if if_reset:
        train_tracker.reset()
        test_tracker.reset()
    else:
        logging.info("WARNING: train_tracker and test_tracker are not reset!!!")
    return train_metric_info, test_metric_info

