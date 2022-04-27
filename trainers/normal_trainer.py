import copy
import logging
import time

import torch
import wandb
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from torch.distributions import Categorical

from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from fedml_core.trainer.model_trainer import ModelTrainer

from utils.data_utils import (
    get_data,
    get_named_data,
    get_all_bn_params,
    apply_gradient,
    clear_grad,
    get_name_params_difference,
    get_local_num_iterations,
    get_avg_num_iterations,
    check_device,
    get_train_batch_data
)



from utils.context import (
    raise_error_without_process,
)

from utils.checkpoint import (
    setup_checkpoint_config
)



from trainers.averager import Averager



class NormalTrainer(ModelTrainer):
    def __init__(self, model, device, criterion, optimizer, lr_scheduler, args, **kwargs):
        super().__init__(model)

        if kwargs['role'] == 'server':
            if "server_index" in kwargs:
                self.server_index = kwargs["server_index"]
            else:
                self.server_index = args.server_index
            self.client_index = None
            self.index = self.server_index

        elif kwargs['role'] == 'client':
            if "client_index" in kwargs:
                self.client_index = kwargs["client_index"]
            else:
                self.client_index = args.client_index
            self.server_index = None
            self.index = self.client_index
        else:
            raise NotImplementedError

        self.role = kwargs['role']

        self.args = args
        self.model = model
        # self.model.to(device)
        self.device = device
        self.criterion = criterion.to(device)
        self.optimizer = optimizer

        self.save_checkpoints_config = setup_checkpoint_config(self.args)

        # For future use
        self.param_groups = self.optimizer.param_groups
        with raise_error_without_process():
            self.param_names = list(
                enumerate([group["name"] for group in self.param_groups])
            )

        self.named_parameters = list(self.model.named_parameters())

        if len(self.named_parameters) > 0:
            self._parameter_names = {v: k for k, v
                                    in sorted(self.named_parameters)}
            #print('Sorted named_parameters')
        else:
            self._parameter_names = {v: 'noname.%s' % i
                                    for param_group in self.param_groups
                                    for i, v in enumerate(param_group['params'])}

        self.averager = Averager(self.args, self.model)

        self.lr_scheduler = lr_scheduler


    def track(self, tracker, summary_n_samples, model, loss, end_of_epoch,
            checkpoint_extra_name="centralized",
            things_to_track=[]):

        logging.debug(f"things_to_track has : {things_to_track}, end_of_epoch: {end_of_epoch}")

        if 'losses_track' in things_to_track:
            assert self.args.losses_track
            tracker.update_local_record(
                'losses_track',
                server_index=self.server_index,
                client_index=self.client_index,
                summary_n_samples=summary_n_samples,
                args=self.args,
                losses=loss.item()
            )


    def epoch_init(self):
        pass

    def epoch_end(self):
        pass

    def update_state(self, **kwargs):
        self.update_loss_state(**kwargs)
        self.update_optimizer_state(**kwargs)

    def update_loss_state(self, **kwargs):
        if self.args.loss_fn in ["LDAMLoss", "FocalLoss"]:
            kwargs['cls_num_list'] = kwargs["selected_cls_num_list"]
            self.criterion.update(**kwargs)
        elif self.args.loss_fn in ["local_FocalLoss", "local_LDAMLoss"]:
            kwargs['cls_num_list'] = kwargs["local_cls_num_list_dict"][self.index]
            self.criterion.update(**kwargs)



    def update_optimizer_state(self, **kwargs):
        pass

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        # for name, param in model_parameters.items():
        #     logging.info(f"Getting params as model_parameters: name:{name}, shape: {param.shape}")
        self.model.load_state_dict(model_parameters)

    def get_model_bn(self):
        all_bn_params = get_all_bn_params(self.model)
        return all_bn_params

    def set_model_bn(self, all_bn_params):
        # logging.info(f"all_bn_params.keys(): {all_bn_params.keys()}")
        # for name, params in all_bn_params.items():
            # logging.info(f"name:{name}, params.shape: {params.shape}")
        for module_name, module in self.model.named_modules():
            if type(module) is nn.BatchNorm2d:
                # logging.info(f"module_name:{module_name}, params.norm: {module.weight.data.norm()}")
                module.weight.data = all_bn_params[module_name+".weight"] 
                module.bias.data = all_bn_params[module_name+".bias"] 
                module.running_mean = all_bn_params[module_name+".running_mean"] 
                module.running_var = all_bn_params[module_name+".running_var"] 
                module.num_batches_tracked = all_bn_params[module_name+".num_batches_tracked"] 


    def get_model_grads(self):
        named_grads = get_named_data(self.model, mode='GRAD')
        # logging.info(f"Getting grads as named_grads: {named_grads}")
        return named_grads

    def set_grad_params(self, named_grads):
        # pass
        self.model.train()
        self.optimizer.zero_grad()
        for name, parameter in self.model.named_parameters():
            parameter.grad.copy_(named_grads[name].data.to(self.device))

    def clear_grad_params(self):
        self.optimizer.zero_grad()

    def update_model_with_grad(self):
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

    # def lr_schedule(self, epoch):
    #     self.lr_scheduler.step(epoch)


    def lr_schedule(self, progress):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(progress)
        else:
            logging.info("No lr scheduler...........")


    def warmup_lr_schedule(self, iterations):
        if self.lr_scheduler is not None:
            self.lr_scheduler.warmup_step(iterations)


    # Used for single machine training
    def train(self, train_data, device, args, **kwargs):
        model = self.model

        model.train()

        epoch_loss = []
        for epoch in range(args.max_epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                # logging.info(images.shape)
                x, labels = x.to(device), labels.to(device)
                self.optimizer.zero_grad()

                if self.args.model_out_feature:
                    output, feat = model(x)
                else:
                    output = model(x)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
                logging.info('Training Epoch: {} iter: {} \t Loss: {:.6f}'.format(
                                epoch, batch_idx, loss.item()))
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(Trainer_ID {}. Train Epo: {} \tLoss: {:.6f}'.format(
                    self.index, epoch, sum(epoch_loss) / len(epoch_loss)))
            self.lr_scheduler.step(epoch=epoch + 1)


    def get_train_batch_data(self, train_local):
        try:
            train_batch_data = self.train_local_iter.next()
            # logging.debug("len(train_batch_data[0]): {}".format(len(train_batch_data[0])))
            # if len(train_batch_data[0]) < self.args.batch_size:
            #     logging.debug("WARNING: len(train_batch_data[0]): {} < self.args.batch_size: {}".format(
            #         len(train_batch_data[0]), self.args.batch_size))
                # logging.debug("train_batch_data[0]: {}".format(train_batch_data[0]))
                # logging.debug("train_batch_data[0].shape: {}".format(train_batch_data[0].shape))
        except:
            self.train_local_iter = iter(train_local)
            train_batch_data = self.train_local_iter.next()
        return train_batch_data


    def summarize(self, model, output, labels,
        tracker, metrics,
        loss,
        epoch, batch_idx,
        mode='train',
        checkpoint_extra_name="centralized",
        things_to_track=[],
        if_update_timer=False,
        train_data=None, train_batch_data=None,
        end_of_epoch=None,
    ):
        if np.isnan(loss.item()):
            logging.info('(WARNING!!!!!!!! Trainer_ID {}. Train epoch: {},\
                iteration: {}, loss is nan!!!! '.format(
                self.index, epoch, batch_idx))
            loss.data.fill_(100)
        metric_stat = metrics.evaluate(loss, output, labels)
        tracker.update_metrics(
            metric_stat, 
            metrics_n_samples=labels.size(0)
        )

        if len(things_to_track) > 0:
            if end_of_epoch is not None:
                pass
            else:
                end_of_epoch = (batch_idx == len(train_data) - 1)
            self.track(tracker, self.args.batch_size, model, loss, end_of_epoch,
                    checkpoint_extra_name=checkpoint_extra_name,
                    things_to_track=things_to_track)

        if if_update_timer:
            """
                Now, the client timer need to be updated by each iteration, 
                so as to record the track infomation.
                But only for epoch training, because One-step training will be scheduled by client or server
            """
            tracker.timer.past_iterations(iterations=1)

        if mode == 'train':
            logging.info('Trainer {}. Glob comm round: {}, Train Epo: {}, iter: {} '.format(
                self.index, tracker.timer.global_comm_round_idx, epoch, batch_idx) + metrics.str_fn(metric_stat))
                # logging.info('(Trainer_ID {}. Local Training Epoch: {}, Iter: {} \tLoss: {:.6f} ACC1:{}'.format(
                #     self.index, epoch, batch_idx, sum(batch_loss) / len(batch_loss), metric_stat['Acc1']))
        elif mode == 'test':
            logging.info('(Trainer_ID {}. Test epoch: {}, iteration: {} '.format(
                self.index, epoch, batch_idx) + metrics.str_fn(metric_stat))
        else:
            raise NotImplementedError
        return metric_stat



    def train_one_epoch(self, train_data=None, device=None, args=None, epoch=0,
                        tracker=None, metrics=None,
                        local_iterations=None,
                        move_to_gpu=True, make_summary=True,
                        clear_grad_bef_opt=True, clear_grad_aft_opt=True,
                        checkpoint_extra_name="centralized",
                        things_to_track=[],
                        **kwargs):
        model = self.model

        if move_to_gpu:
            model.to(device)
        model.train()
        batch_loss = []
        if local_iterations is None:
            iterations = len(train_data)
        else:
            iterations = local_iterations

        if self.args.fedprox:
            # previous_model = copy.deepcopy(self.trainer.get_model_params())
            previous_model = kwargs["previous_model"]

        # for batch_idx, (x, labels) in enumerate(train_data):
        for batch_idx in range(iterations):
            train_batch_data = self.get_train_batch_data(train_data)
            x, labels = train_batch_data
            # if batch_idx > 5:
            #     break

            real_batch_size = x.shape[0]
            x, labels = x.to(device), labels.to(device)
            if clear_grad_bef_opt:
                self.optimizer.zero_grad()

            output = model(x)

            loss = self.criterion(output, labels)

            if self.args.fedprox:
                fed_prox_reg = 0.0
                previous_model = kwargs["previous_model"]
                for name, param in model.named_parameters():
                    fed_prox_reg += ((self.args.fedprox_mu / 2) * \
                        torch.norm((param - previous_model[name].data.to(device)))**2)
                loss += fed_prox_reg

            loss.backward()
            self.optimizer.step()

            batch_loss.append(loss.item())

            logging.debug(f"epoch: {epoch}, Loss is {loss.item()}")

            if make_summary and (tracker is not None) and (metrics is not None):
                self.summarize(model, output, labels,
                        tracker, metrics,
                        loss,
                        epoch, batch_idx,
                        mode='train',
                        checkpoint_extra_name=checkpoint_extra_name,
                        things_to_track=things_to_track,
                        if_update_timer=True if self.args.record_dataframe else False,
                        train_data=train_data, train_batch_data=train_batch_data,
                        end_of_epoch=None,
                    )



    def train_one_step(self, train_batch_data, device=None, args=None,
            epoch=None, iteration=None, end_of_epoch=False,
            tracker=None, metrics=None,
            move_to_gpu=True, make_summary=True,
            clear_grad_bef_opt=True, clear_grad_aft_opt=True,
            checkpoint_extra_name="centralized",
            things_to_track=[],
            **kwargs
        ):

        model = self.model

        if move_to_gpu:
            model.to(device)

        model.train()

        x, labels = train_batch_data
        x, labels = x.to(device), labels.to(device)
        real_batch_size = x.shape[0]

        if clear_grad_bef_opt:
            self.optimizer.zero_grad()

        output = model(x)
        loss = self.criterion(output, labels)

        loss.backward()
        self.optimizer.step()

        if make_summary and (tracker is not None) and (metrics is not None):
            self.summarize(model, output, labels,
                    tracker, metrics,
                    loss,
                    epoch, iteration,
                    mode='train',
                    checkpoint_extra_name=checkpoint_extra_name,
                    things_to_track=things_to_track,
                    if_update_timer=False,
                    train_data=None, train_batch_data=train_batch_data,
                    end_of_epoch=end_of_epoch,
                )

        return loss, output, labels


    def infer_bw_one_step(self, train_batch_data, device=None, args=None,
            epoch=None, iteration=None, end_of_epoch=False,
            tracker=None, metrics=None,
            move_to_gpu=True, model_train=True, make_summary=True,
            clear_grad_bef_opt=True, clear_grad_aft_opt=True,
            checkpoint_extra_name="centralized",
            things_to_track=[],
            **kwargs
        ):
        """
            inference and BP without optimization
        """
        model = self.model

        if move_to_gpu:
            model.to(device)

        if model_train:
            model.train()
        else:
            model.eval()

        x, labels = train_batch_data
        x, labels = x.to(device), labels.to(device)

        if clear_grad_bef_opt:
            self.optimizer.zero_grad()

        output = model(x)
        loss = self.criterion(output, labels)

        if make_summary and (tracker is not None) and (metrics is not None):
            self.summarize(model, output, labels,
                    tracker, metrics,
                    loss,
                    epoch, iteration,
                    mode='train',
                    checkpoint_extra_name=checkpoint_extra_name,
                    things_to_track=things_to_track,
                    if_update_timer=False,
                    train_data=None, train_batch_data=train_batch_data,
                    end_of_epoch=end_of_epoch,
                )

        return loss, output, labels



    def test(self, test_data, device=None, args=None, epoch=None,
            tracker=None, metrics=None,
            move_to_gpu=True, make_summary=True,
            clear_grad_bef_opt=True, clear_grad_aft_opt=True,
            checkpoint_extra_name="centralized",
            things_to_track=[],
            **kwargs):

        model = self.model
        Acc_accm = 0.0

        model.eval()
        if move_to_gpu:
            model.to(device)
        with torch.no_grad():
            for batch_idx, (x, labels) in enumerate(test_data):
                x = x.to(device)
                labels = labels.to(device)
                real_batch_size = x.shape[0]

                output = model(x)

                loss = self.criterion(output, labels)

                if make_summary and (tracker is not None) and (metrics is not None):
                    metric_stat = self.summarize(model, output, labels,
                            tracker, metrics,
                            loss,
                            epoch, batch_idx,
                            mode='test',
                            checkpoint_extra_name=checkpoint_extra_name,
                            things_to_track=things_to_track,
                            if_update_timer=False,
                            train_data=test_data, train_batch_data=None,
                            end_of_epoch=False,
                        )
                    logging.debug(f"metric_stat[Acc1] is {metric_stat['Acc1']} ")
                    Acc_accm += metric_stat["Acc1"]
            logging.debug(f"Total is {Acc_accm} , averaged is {Acc_accm / (batch_idx+1)}")



    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None,
                        epoch=None, iteration=None, tracker=None, metrics=None):
        pass














