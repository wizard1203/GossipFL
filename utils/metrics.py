import math

import torch

class Metrics(object):
    """
        Keep this class stateless.
    """

    def __init__(self, topks=[1], task="classification"):
        self.task = task
        self.topks = topks
        self.get_callbacks(topks, task)
        # self.metric_names = self.get_metric_names(topks, task)
        # self.metrics_fn = self._get_metric_measure(topks, task)
        # self.get_metric_info_func = self._get_get_metric_info_func()
        # self.str_fn = self._get_str_fn(task)


    def evaluate(self, loss, output, target, **kargs):
        return self.metrics_fn(loss, output, target, **kargs)


    def get_callbacks(self, topks, task):
        if task == "classification":
            self.metric_names = ["Acc{}".format(topk) for topk in topks]
            self.metric_names += ["Loss"]
            self.metrics_fn = self._classification_metric
            self.get_metric_info_func = None
            self.str_fn = self._classification_str_fn
        elif task == "stackoverflow_lr":
            self.metric_names = ["Acc", "Loss", "Precision", "Recall"]
            self.metrics_fn = self._stackoverflow_lr_metric
            self.get_metric_info_func = None
            self.str_fn = self._stackoverflow_str_fn
        elif task == "ptb":
            self.metric_names = ["Loss"]
            self.metrics_fn = self._ptb_metric
            self.get_metric_info_func = self._get_ptb_metric_info
            self.str_fn = self._ptb_str_fn
        else:
            raise NotImplementedError



    # @classmethod
    # def get_metric_names(cls, topks, task):
    #     if task == "classification":
    #         metric_names = ["Acc{}".format(topk) for topk in topks]
    #         metric_names += ["Loss"]
    #     elif task == "stackoverflow_lr":
    #         metric_names = ["Acc", "Loss", "Precision", "Recall"]
    #     else:
    #         raise NotImplementedError
    #     return metric_names

    # def _get_metric_measure(self, topks, task):
    #     if task == "classification":
    #         return self._classification_metric
    #     elif task == "stackoverflow_lr":
    #         return self._stackoverflow_lr_metric
    #     elif task == "ptb":
    #         return self._ptb_metric
    #     else:
    #         raise NotImplementedError
    #     assert self.metric_names is not None

    # def _get_str_fn(self, task):
    #     if task == "classification":
    #         return self._classification_str_fn
    #     elif task == "stackoverflow_lr":
    #         return self._stackoverflow_str_fn
    #     elif task == "ptb":
    #         return self._ptb_str_fn
    #     else:
    #         raise NotImplementedError

    def _classification_metric(self, loss, output, target, **kargs):
        """Computes the precision@k for the specified values of k"""
        metric_stat = {}
        metric_stat["Loss"] = loss.item()

        maxk = max(self.topks)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        for topk in self.topks:
            correct_k = correct[:topk].view(-1).float().sum(0, keepdim=True)
            # res.append(correct_k.mul_(100.0 / batch_size).item())
            metric_stat["Acc{}".format(topk)] = correct_k.mul_(100.0 / batch_size).item()

        return metric_stat

    def _stackoverflow_lr_metric(self, loss, output, target, **kargs):
        metric_stat = {}
        metric_stat["Loss"] = loss.item()
        predicted = (output > .5).int()
        correct = predicted.eq(target).sum(axis=-1).eq(target.size(1)).sum()
        true_positive = ((target * predicted) > .1).int().sum(axis=-1)
        metric_stat["Precision"] = true_positive / (predicted.sum(axis=-1) + 1e-13)
        metric_stat["Recall"] = true_positive / (target.sum(axis=-1) + 1e-13)
        metric_stat["Acc"] = correct.mul_(100.0 / target.size(0)).item()
        metric_stat["Loss"] = loss.item()
        return metric_stat

    def _ptb_metric(self, loss, output, target, **kargs):
        # avg_loss = kargs["avg_loss"]
        metric_stat = {}
        metric_stat["Loss"] = loss.item()
        return metric_stat

    """
        stat: dict((name, AverageMeter())
    """
    def _get_ptb_metric_info(self, stat, **kargs):
        metric_info = dict((name, val.avg) for name, val in stat.items())
        metric_info['ppl'] = math.exp(metric_info['Loss'])
        return metric_info


    def _classification_str_fn(self, metric_info):
        base = "Loss: {}".format(metric_info["Loss"])
        for topk in self.topks:
            base += ", Acc{}: {}".format(topk, metric_info["Acc{}".format(topk)])
        base += "."
        return base


    def _stackoverflow_str_fn(self):
        pass

    def _ptb_str_fn(self, metric_info):
        base = "Loss: {}".format(metric_info["Loss"])
        return base



