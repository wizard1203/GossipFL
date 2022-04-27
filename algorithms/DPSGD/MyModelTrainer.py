import logging

import torch
from torch import nn
import wandb

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
        # Now only support sgd
        assert args.client_optimizer == "sgd"

        self.optimizer = create_optimizer(args, self.model)
        self.lr_scheduler = create_scheduler(args, self.optimizer)
        self.lr_scheduler.step(0)

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def lr_schedule(self, epoch):
        self.lr_scheduler.step(epoch)


    def train_one_step(self, train_batch_data, device, args, tracker=None, metrics=None):
        model = self.model

        model.to(device)
        model.train()
        x, labels = train_batch_data
        x, labels = x.to(device), labels.to(device)
        self.optimizer.zero_grad()
        output = model(x)
        loss = self.criterion(output, labels)
        loss.backward()
        self.optimizer.step()
        if (tracker is not None) and (metrics is not None): 
            metric_stat = metrics.evaluate(loss, output, labels)
            tracker.update_metrics(metric_stat, n_samples=labels.size(0))

        return loss, output, labels


    # not used
    def train(self, train_data, device, args):
        pass

    def test(self, test_data, device, args, tracker=None, metrics=None):
        model = self.model

        model.eval()
        model.to(device)
        with torch.no_grad():
            for batch_idx, (x, labels) in enumerate(test_data):
                x = x.to(device)
                labels = labels.to(device)
                output = model(x)
                # logging.debug("labels: {}".format(labels))
                # logging.debug("output: {}".format(output))
                loss = self.criterion(output, labels)
                if (metrics is not None) and (tracker is not None):
                    metric_stat = metrics.evaluate(loss, output, labels)
                    tracker.update_metrics(metric_stat, n_samples=labels.size(0))
                    logging.info('(Trainer_ID {}. Testing Iter: {} \tLoss: {:.6f} ACC1:{}'.format(
                        self.id, batch_idx, loss.item(), metric_stat['Acc1']))
                else:
                     logging.info('(Trainer_ID {}. Testing Iter: {} \tLoss: {:.6f}'.format(
                        self.id, batch_idx, loss.item()))


    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None):
        pass
