import logging

import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer, required


class FedProx(Optimizer):
    r"""Implements FedAvg and FedProx. Local Solver can have momentum.
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        mu (float): parameter for proximal local SGD
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v
        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
             v = \rho * v + lr * g \\
             p = p - v
        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, variance=0, mu=0):
        
        # self.gmf = gmf
        # self.ratio = ratio
        # self.itr = 0
        # self.a_sum = 0
        self.mu = mu

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, variance=variance)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(FedProx, self).__init__(params, defaults)


    def __setstate__(self, state):
        super(FedProx, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                
                param_state = self.state[p]
                if 'old_init' not in param_state:
                    param_state['old_init'] = torch.clone(p.data).detach()
                    param_state['old_init'] = param_state['old_init'].to(p.data.device)

                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # logging.info("p.data.device: {}, param_state['old_init'].deivce: {}".format(
                #     p.data.device, param_state['old_init'].get_device()
                # ))
                # apply proximal update
                param_state['old_init'] = param_state['old_init'].to(p.data.device)
                d_p.add_(self.mu, p.data - param_state['old_init'])
                p.data.add_(-group['lr'], d_p)
        return loss

    def clear_buffer(self):
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                # Reinitialize momentum buffer
                if 'momentum_buffer' in param_state:
                    param_state['momentum_buffer'].zero_()

    def update_old_init(self):
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['old_init'] = torch.clone(p.data).detach()
                param_state['old_init'] = param_state['old_init'].to(p.data.device)

    # def average(self):
    #     param_list = []
    #     for group in self.param_groups:
    #         for p in group['params']:
    #             p.data.mul_(self.ratio)
    #             param_list.append(p.data)

    #     communicate(param_list, dist.all_reduce)

    #     for group in self.param_groups:
    #         for p in group['params']:
    #             param_state = self.state[p]
    #             param_state['old_init'] = torch.clone(p.data).detach()
    #             # Reinitialize momentum buffer
    #             if 'momentum_buffer' in param_state:
    #                 param_state['momentum_buffer'].zero_()

