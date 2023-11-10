import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

from utils.data_utils import get_per_cls_weights


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, cls_num_list=None, gamma=0., imbalance_beta=0.9999, args=None):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.args = args
        self.gamma = gamma
        self.imbalance_beta = imbalance_beta
        if self.args.imbalance_loss_reweight:
            self.weight = get_per_cls_weights(cls_num_list, imbalance_beta)
        else:
            self.weight = None

    def update(self, **kwargs):
        if self.args.imbalance_loss_reweight:
            if "cls_num_list" in kwargs and kwargs["cls_num_list"] is not None:
                if "imbalance_beta" in kwargs and kwargs["imbalance_beta"] is not None:
                    self.weight = get_per_cls_weights(kwargs["cls_num_list"], kwargs["imbalance_beta"])
                else:
                    self.weight = get_per_cls_weights(kwargs["cls_num_list"], self.imbalance_beta)
            else:
                pass
        else:
            logging.info("WARNING: the imbalance weight has not been updated.")
            self.weight = None

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)

class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list=None, max_m=0.5, s=30, imbalance_beta=0.9999, args=None):
        super(LDAMLoss, self).__init__()
        self.args = args
        self.imbalance_beta = imbalance_beta
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        # m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        self.max_m = max_m
        assert s > 0
        self.s = s
        if self.args.imbalance_loss_reweight:
            self.weight = get_per_cls_weights(cls_num_list, imbalance_beta)
        else:
            self.weight = None

    def update(self, **kwargs):
        if self.args.imbalance_loss_reweight:
            if "cls_num_list" in kwargs and kwargs["cls_num_list"] is not None:
                if "imbalance_beta" in kwargs and kwargs["imbalance_beta"] is not None:
                    self.weight = get_per_cls_weights(kwargs["cls_num_list"], kwargs["imbalance_beta"])
                else:
                    self.weight = get_per_cls_weights(kwargs["cls_num_list"], self.imbalance_beta)
            else:
                pass
        else:
            logging.info("WARNING: the imbalance weight has not been updated.")
            self.weight = None

        if "cls_num_list" in kwargs and kwargs["cls_num_list"] is not None:
            cls_num_list = kwargs["cls_num_list"]
            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (self.max_m / np.max(m_list))
            # m_list = torch.cuda.FloatTensor(m_list)
            self.m_list = m_list
        else:
            pass

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.type(torch.cuda.FloatTensor)
        self.m_list = torch.cuda.FloatTensor(self.m_list).to(device=x.device)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)



    # def forward(self, x, target, m_list=None, weight=None):
    #     index = torch.zeros_like(x, dtype=torch.uint8)
    #     index.scatter_(1, target.data.view(-1, 1), 1)
        
    #     index_float = index.type(torch.cuda.FloatTensor)
    #     if m_list is not None:
    #         batch_m = torch.matmul(m_list[None, :], index_float.transpose(0,1))
    #     else:
    #         batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))

    #     if weight is not None:
    #         input_weight = weight
    #     else:
    #         input_weight = self.weight

    #     batch_m = batch_m.view((-1, 1))
    #     x_m = x - batch_m
    
    #     output = torch.where(index, x_m, x)
    #     return F.cross_entropy(self.s*output, target, weight=input_weight)






