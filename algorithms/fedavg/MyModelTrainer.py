import logging
import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_core.trainer.model_trainer import ModelTrainer



from optim.build import create_optimizer
from loss_fn.build import create_loss
from lr_scheduler.build import create_scheduler

class MyModelTrainer(ModelTrainer):
    def __init__(self, model, device, args):
        super().__init__(model)

        self.model = model
        self.args = args
        self.criterion = nn.CrossEntropyLoss().to(device)
        # In fedavg, decay according to the round

        self.optimizer = create_optimizer(args, self.model)
        self.lr_scheduler = create_scheduler(args, self.optimizer)
        self.lr_scheduler.step(0)

        self.train_loss_fn = nn.CrossEntropyLoss().to(device)
        self.validate_loss_fn = nn.CrossEntropyLoss().to(device)


    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        pass

    def fedavg_train(self, train_data, device, round_idx, args):
        model = self.model

        model.to(device)
        model.train()

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                # logging.debug(images.shape)
                x, labels = x.to(device), labels.to(device)
                self.optimizer.zero_grad()
                log_probs = model(x)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
                logging.info('Local Training Epoch: {} iter: {} \t Loss: {:.6f}'.format(
                                epoch, batch_idx, loss.item()))
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(Trainer_ID {}. Local Training Epoch: {} \tLoss: {:.6f}'.format(self.id,
                                                                                              epoch,
                                                                                              sum(epoch_loss) / len(
                                                                                                  epoch_loss)))
        self.lr_scheduler.step(epoch=args.global_round_idx)

    def test(self, test_data, device, args, tracker=None, metrics=None):
        """
            test 
        """
        model = self.model

        model.eval()
        model.to(device)

        with torch.no_grad():
            x, labels = test_data
            x, labels = x.to(device), labels.to(device)
            self.optimizer.zero_grad()
            log_probs = model(x)
            loss = self.criterion(log_probs, labels)
            if (tracker is not None) and (metrics is not None): 
                metric_stat = metrics.evaluate(loss, log_probs, labels)
                tracker.update_metrics(metric_stat, n_samples=labels.size(0))
            return loss, log_probs, labels


    # def test(self, test_data, device, args):
    #     model = self.model

    #     model.eval()
    #     model.to(device)

    #     metrics = {
    #         'test_correct': 0,
    #         'test_loss': 0,
    #         'test_precision': 0,
    #         'test_recall': 0,
    #         'test_total': 0
    #     }

    #     criterion = nn.CrossEntropyLoss().to(device)
    #     with torch.no_grad():
    #         for batch_idx, (x, target) in enumerate(test_data):
    #             x = x.to(device)
    #             target = target.to(device)
    #             pred = model(x)
    #             loss = criterion(pred, target)
    #             if args.dataset == "stackoverflow_lr":
    #                 predicted = (pred > .5).int()
    #                 correct = predicted.eq(target).sum(axis=-1).eq(target.size(1)).sum()
    #                 true_positive = ((target * predicted) > .1).int().sum(axis=-1)
    #                 precision = true_positive / (predicted.sum(axis=-1) + 1e-13)
    #                 recall = true_positive / (target.sum(axis=-1) + 1e-13)
    #                 metrics['test_precision'] += precision.sum().item()
    #                 metrics['test_recall'] += recall.sum().item()
    #             else:
    #                 _, predicted = torch.max(pred, -1)
    #                 correct = predicted.eq(target).sum()

    #             metrics['test_correct'] += correct.item()
    #             metrics['test_loss'] += loss.item() * target.size(0)
    #             metrics['test_total'] += target.size(0)
    #             logging.info('Local Testing iter: {} \t Loss: {:.6f} Acc: {:.6f}'.format(
    #                             batch_idx, loss.item(),  100.0*metrics['test_correct']/metrics['test_total']))

    #     return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
