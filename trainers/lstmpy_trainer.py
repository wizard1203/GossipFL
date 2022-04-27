import copy
import logging
import time

import torch
import wandb
from torch import nn
import numpy as np


from fedml_core.trainer.model_trainer import ModelTrainer


from utils.data_utils import (
    get_data,
    get_named_data,
    apply_gradient
)
from utils.context import (
    raise_error_without_process,
    get_lock,
)
import model.nlp.lstm as lstmpy


class LSTMTrainer(ModelTrainer):
    def __init__(self, model, device, criterion, optimizer, lr_scheduler, args, **kwargs):
        super().__init__(model)
        if "client_index" in kwargs:
            client_index = kwargs["client_index"]
        else:
            client_index = args.client_index
        self.args = args
        self.model = model
        self.model.to(device)
        self.device = device
        self.criterion = criterion.to(device)
        self.optimizer = optimizer

        # For future use
        self.param_groups = self.optimizer.param_groups
        with raise_error_without_process():
            self.param_names = list(
                enumerate([group["name"] for group in self.param_groups])
            )

        self.named_parameters = list(self.model.named_parameters())

        if len(self.named_parameters) > 0:
            self._parameter_names = {
                v: k for k, v
                in sorted(self.named_parameters)
            }
            #print('Sorted named_parameters')
        else:
            self._parameter_names = {
                v: 'noname.%s' % i
                for param_group in self.param_groups
                for i, v in enumerate(param_group['params'])
            }

        self.lr_scheduler = lr_scheduler
        # if self.lr_scheduler is not None:
        #     self.lr_scheduler.step(0)
        # ===========================================================================
        self.hidden = None



    def init_hidden(self):
        self.hidden = self.model.init_hidden()
        # if self.hidden is None:
        #     raise RuntimeError

    def epoch_init(self):
        self.init_hidden()

    def epoch_end(self):
        pass

    def update_state(self, **kwargs):
        self.update_loss_state(**kwargs)
        self.update_optimizer_state(**kwargs)


    def update_loss_state(self, **kwargs):
        if self.args.loss_fn in ["LDAMLoss", "FocalLoss"]:
            self.criterion.update(**kwargs)
        elif self.args.loss_fn in ["local_FocalLoss", "local_LDAMLoss"]:
            kwargs['cls_num_list'] == None
            self.criterion.update(**kwargs)



    def update_optimizer_state(self, **kwargs):
        pass

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)


    def get_model_grads(self):
        named_grads = get_named_data(self.model, mode='GRAD')
        return named_grads

    def set_grad_params(self, named_grads):
        # pass
        # self.model.train()
        self.optimizer.zero_grad()
        # # for name, grad in named_grads:
        # #     # if name not in self.param_names or len(grad.size()) ==0:
        # #     if name not in self._parameter_names:
        # #         continue
        for name, parameter in self.model.named_parameters():
            # logging.info("name: {}, parameter: {}, parameter.grad: {}".format(
            #     name, parameter, parameter.grad
            # ))
            # if name not in named_grads:
            #     continue
            # parameter.grad.data = named_grads[name].data.to(self.device)
            # parameter.grad = named_grads[name].data.to(self.device)
            parameter.grad.copy_(named_grads[name].data.to(self.device))
        # self.optimizer.zero_grad()


    def clear_grad_params(self):
        self.optimizer.zero_grad()

    def update_model_with_grad(self):
        if self.args.lstm_clip_grad:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.lstm_clip_grad_thres)
        self.model.to(self.device)
        self.optimizer.step()

    def get_optim_state(self):
        return self.optimizer.state


    def clear_optim_buffer(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.optimizer.state[p]
                # Reinitialize momentum buffer
                if 'momentum_buffer' in param_state:
                    param_state['momentum_buffer'].zero_()


    def lr_schedule(self, progress):
        self.lr_scheduler.step(progress)

    def warmup_lr_schedule(self, iterations):
        self.lr_scheduler.warmup_step(iterations)


    # Used for single machine training
    def train(self, train_data, device, args, **kwargs):
        model = self.model
        model.to(device)
        model.train()

        epoch_loss = []
        for epoch in range(args.max_epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                # logging.info(images.shape)
                x, labels = x.to(device), labels.to(device)
                self.optimizer.zero_grad()
                if self.hidden is not None:
                    self.hidden = lstmpy.repackage_hidden(self.hidden)
                output, self.hidden = model(x, self.hidden)
                # loss = self.criterion(output, labels)
                if self.args.model == 'lstm':
                    tt = torch.squeeze(labels.view(-1, self.args.batch_size * self.args.lstm_num_steps))
                    loss = self.criterion(output.view(-1, self.model.vocab_size), tt)
                elif self.args.model == 'rnn':
                    loss = self.criterion(output, labels)
                else:
                    raise NotImplementedError
                loss.backward()
                if self.args.lstm_clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.lstm_clip_grad_thres)
                self.optimizer.step()
                batch_loss.append(loss.item())
                logging.info('Training Epoch: {} iter: {} \t Loss: {:.6f}'.format(
                                epoch, batch_idx, loss.item()))
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(Trainer_ID {}. Training Epoch: {} \tLoss: {:.6f}'.format(
                    self.id, epoch, sum(epoch_loss) / len(epoch_loss)))
            self.lr_scheduler.step(epoch=epoch + 1)



    def train_one_epoch(self, train_data, device, args, epoch, tracker=None, metrics=None, **kwargs):
        model = self.model
        model.to(device)
        model.train()
        batch_loss = []
        for batch_idx, (x, labels) in enumerate(train_data):
            x, labels = x.to(device), labels.to(device)
            self.optimizer.zero_grad()
            if self.hidden is not None:
                self.hidden = lstmpy.repackage_hidden(self.hidden)
            self.hidden = (self.hidden[0].to(device), self.hidden[1].to(device))
            output, self.hidden = model(x, self.hidden)

            # logging.debug("labels: {}".format(labels))
            # logging.debug("pred: {}".format(output))
            # loss = self.criterion(output, labels)
            if self.args.model == 'lstm':
                tt = torch.squeeze(labels.view(-1, self.args.batch_size * self.args.lstm_num_steps))
                loss = self.criterion(output.view(-1, self.model.vocab_size), tt)
            elif self.args.model == 'rnn':
                loss = self.criterion(output, labels)
            else:
                raise NotImplementedError
            if self.args.lstm_clip_grad:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.lstm_clip_grad_thres)
            loss.backward()
            self.optimizer.step()
            batch_loss.append(loss.item())
            if (tracker is not None) and (metrics is not None):
                if np.isnan(loss.item()):
                    logging.info('(WARNING!!!!!!!! Trainer_ID {}. Train epoch: {},\
                        iteration: {}, loss is nan!!!! '.format(
                        self.id, epoch, batch_idx))
                    loss.data.fill_(100)
                metric_stat = metrics.evaluate(loss, output, labels)
                tracker.update_metrics(metric_stat, n_samples=labels.size(0))
                logging.info('(Trainer_ID {}. Training Epoch: {}, Iter: {} '.format(
                    self.id, epoch, batch_idx) + metrics.str_fn(metric_stat))
                    # logging.info('(Trainer_ID {}. Local Training Epoch: {}, Iter: {} \tLoss: {:.6f} ACC1:{}'.format(
                    #     self.id, epoch, batch_idx, sum(batch_loss) / len(batch_loss), metric_stat['Acc1']))
            else:
                pass



    def train_one_step(self, train_batch_data, device, args, epoch=None, iteration=None, tracker=None, metrics=None, **kwargs):
        model = self.model
        model.to(device)
        model.train()
        x, labels = train_batch_data
        x, labels = x.to(device), labels.to(device)
        self.optimizer.zero_grad()
        if self.hidden is not None:
            self.hidden = lstmpy.repackage_hidden(self.hidden)
        # logging.info("x on : {}".format(x.device))
        # logging.info("self.hidden: {}".format(self.hidden))
        # logging.info("self.hidden on :{}".format(self.hidden.device))
        output, self.hidden = model(x, self.hidden)
        # loss = self.criterion(output, labels)
        if self.args.model == 'lstm':
            tt = torch.squeeze(labels.view(-1, self.args.batch_size * self.args.lstm_num_steps))
            loss = self.criterion(output.view(-1, self.model.vocab_size), tt)
        elif self.args.model == 'rnn':
            loss = self.criterion(output, labels)
        else:
            raise NotImplementedError
        loss.backward()
        if self.args.lstm_clip_grad:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.lstm_clip_grad_thres)
        self.optimizer.step()
        if (tracker is not None) and (metrics is not None):
            if np.isnan(loss.item()):
                logging.info('(WARNING!!!!!!!! Trainer_ID {}. Train epoch: {},\
                    iteration: {}, loss is nan!!!! '.format(
                    self.id, epoch, iteration))
                loss.data.fill_(100)
            metric_stat = metrics.evaluate(loss, output, labels)
            tracker.update_metrics(metric_stat, n_samples=labels.size(0))
            logging.info('(Trainer_ID {}. Train epoch: {}, iteration: {} '.format(
                self.id, epoch, iteration) + metrics.str_fn(metric_stat))

        return loss, output, labels


    def infer_bw_one_step(self, train_batch_data, device, args, epoch=None, iteration=None, tracker=None, metrics=None, **kwargs):
        """
            inference and BP without optimization
        """
        model = self.model

        model.to(device)
        model.train()
        x, labels = train_batch_data
        x, labels = x.to(device), labels.to(device)
        self.optimizer.zero_grad()
        # if self.hidden is None:
        #     raise RuntimeError
        # if self.args.model == 'lstm':
        #     self.hidden = lstmpy.repackage_hidden(self.hidden)
        if self.hidden is not None:
            self.hidden = lstmpy.repackage_hidden(self.hidden)
        # logging.info("self.hidden: {}".format(self.hidden))
        # logging.info("self.hidden on :{}".format(self.hidden.device))
        output, self.hidden = model(x, self.hidden)
        # if self.hidden is None:
        #     raise RuntimeError
        # loss = self.criterion(output, labels)
        if self.args.model == 'lstm':
            tt = torch.squeeze(labels.view(-1, self.args.batch_size * self.args.lstm_num_steps))
            loss = self.criterion(output.view(-1, self.model.vocab_size), tt)
        elif self.args.model == 'rnn':
            loss = self.criterion(output, labels)
        else:
            raise NotImplementedError
        loss.backward()
        if self.args.lstm_clip_grad:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.lstm_clip_grad_thres)
        if (tracker is not None) and (metrics is not None): 
            if np.isnan(loss.item()):
                logging.info('(WARNING!!!!!!!! Trainer_ID {}. Train epoch: {},\
                    iteration: {}, loss is nan!!!! '.format(
                    self.id, epoch, iteration))
                loss.data.fill_(100)
            metric_stat = metrics.evaluate(loss, output, labels)
            tracker.update_metrics(metric_stat, n_samples=labels.size(0))
            logging.info('(Trainer_ID {}. Train epoch: {}, iteration: {} '.format(
                self.id, epoch, iteration) + metrics.str_fn(metric_stat))
        return loss, output, labels


    def test(self, test_data, device, args, epoch, tracker=None, metrics=None, **kwargs):
        model = self.model
        model.eval()
        model.to(device)
        with torch.no_grad():
            for batch_idx, (x, labels) in enumerate(test_data):
                x = x.to(device)
                labels = labels.to(device)
                hidden = self.model.init_hidden()
                output, hidden = model(x, hidden)
                # logging.debug("labels: {}".format(labels))
                # logging.debug("output: {}".format(output))
                if self.args.model == 'lstm':
                    tt = torch.squeeze(labels.view(-1, self.args.batch_size * self.args.lstm_num_steps))
                    loss = self.criterion(output.view(-1, self.model.vocab_size), tt)
                elif self.args.model == 'rnn':
                    loss = self.criterion(output, labels)
                else:
                    raise NotImplementedError

                if (tracker is not None) and (metrics is not None):
                    if np.isnan(loss.item()):
                        logging.info('(WARNING!!!!!!!! Trainer_ID {}. Train epoch: {},\
                            iteration: {}, loss is nan!!!! '.format(
                            self.id, epoch, batch_idx))
                        loss.data.fill_(100)
                    # lstm_num_steps = kwargs["ptb_num_steps"]
                    # batch_size = kwargs["batch_size"]
                    if self.args.model == 'lstm':
                        metric_stat = metrics.evaluate(loss, output, labels,
                                                   lstm_num_steps=self.model.num_steps)
                    elif self.args.model == 'rnn':
                        metric_stat = metrics.evaluate(loss, output, labels)
                    else:
                        raise NotImplementedError
                    tracker.update_metrics(metric_stat, n_samples=labels.size(0))
                    logging.info('(Trainer_ID {}. Test epoch: {}, iteration: {} '.format(
                        self.id, epoch, batch_idx) + metrics.str_fn(metric_stat))
                else:
                    pass



    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None,
                           epoch=None, iteration=None, tracker=None, metrics=None):
        pass



