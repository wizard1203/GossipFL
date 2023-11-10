import logging
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch


def get_n_bits(tensor):
    return 8 * tensor.nelement() * tensor.element_size()



"""some aggregation functions."""


def get_params(model, args):
    """
        some features maybe needed in this
    """
    params = [
        {
            "params": [value],
            "name": key,
            "weight_decay": args.wd if "bn" not in key else 0.0,
            "param_size": value.size(),
            "nelement": value.nelement(),
        }
        for key, value in model.named_parameters()
    ]
    return params


def _get_data(param_groups, idx, is_get_grad):
    # Define the function to get the data.
    # when we create the param_group, each group only has one param.
    if is_get_grad:
        return param_groups[idx]["params"][0].grad
    else:
        return param_groups[idx]["params"][0]


def _get_shape(param_groups, idx):
    return param_groups[idx]["param_size"], param_groups[idx]["nelement"]


def get_data(param_groups, param_names, is_get_grad=True):
    data, shapes = [], []
    for idx, _ in param_names:
        _data = _get_data(param_groups, idx, is_get_grad)
        if _data is not None:
            data.append(_data)
            shapes.append(_get_shape(param_groups, idx))
    return data, shapes

def get_named_data(model, mode='MODEL', use_cuda=True):
    """
        getting the whole model and getting the gradients can be conducted
        by using different methods for reducing the communication.
        `model` choices: ['MODEL', 'GRAD', 'MODEL+GRAD'] 
    """
    if mode == 'MODEL':
        own_state = model.cpu().state_dict()
        return own_state
    elif mode == 'GRAD':
        grad_of_params = {}
        for name, parameter in model.named_parameters():
            if use_cuda:
                grad_of_params[name] = parameter.grad
            else:
                grad_of_params[name] = parameter.grad.cpu()
        return grad_of_params
    elif mode == 'MODEL+GRAD':
        model_and_grad = {}
        for name, parameter in model.named_parameters():
            if use_cuda:
                model_and_grad[name] = parameter.data
                model_and_grad[name+b'_gradient'] = parameter.grad
            else:
                model_and_grad[name] = parameter.data.cpu()
                model_and_grad[name+b'_gradient'] = parameter.grad.cpu()
        return model_and_grad 


def average_named_params(named_params_list, sum):
    """
        This is a weighted average operation.
    """
    # logging.info("################aggregate: %d" % len(named_params_list))
    (_, averaged_params) = named_params_list[0]
    for k in averaged_params.keys():
        for i in range(0, len(named_params_list)):
            local_sample_number, local_named_params = named_params_list[i]
            w = local_sample_number / sum
            logging.debug("aggregating ---- local_sample_number/sum: {}/{}, ".format(
                local_sample_number, sum))
            if i == 0:
                averaged_params[k] = (local_named_params[k] * w).type(averaged_params[k].dtype)
            else:
                averaged_params[k] += (local_named_params[k] * w).type(averaged_params[k].dtype)
    return averaged_params

def average_tensors(tensors, weights):
    if isinstance(tensors, list):
        sum = np.sum(weights)
        averaged_tensor = tensors[0]
        for i, tensor in enumerate(tensors):
            w = weights[i] / sum
            if i == 0:
                averaged_tensor = tensor * w
            else:
                averaged_tensor += tensor * w
    elif isinstance(tensors, dict):
        sum = np.sum(list(weights.values()))
        averaged_tensor = None
        for i, key in enumerate(tensors.keys()):
            w = weights[key] / sum
            if i == 0:
                averaged_tensor = tensors[key] * w
            else:
                averaged_tensor += tensors[key] * w
    else:
        raise NotImplementedError()
    return averaged_tensor


"""tensor reshape."""

def flatten(tensors, shapes=None, use_cuda=True):
    # init and recover the shapes vec.
    pointers = [0]
    if shapes is not None:
        for shape in shapes:
            pointers.append(pointers[-1] + shape[1])
    else:
        for tensor in tensors:
            pointers.append(pointers[-1] + tensor.nelement())

    # flattening.
    vec = torch.empty(
        pointers[-1], dtype=tensors[0].dtype,
        device=tensors[0].device if tensors[0].is_cuda and use_cuda else "cpu",
    )

    for tensor, start_idx, end_idx in zip(tensors, pointers[:-1], pointers[1:]):
        vec[start_idx:end_idx] = tensor.data.view(-1)
    return vec


def unflatten(tensors, synced_tensors, shapes):
    pointer = 0

    for tensor, shape in zip(tensors, shapes):
        param_size, nelement = shape
        tensor.data[:] = synced_tensors[pointer : pointer + nelement].view(param_size)
        pointer += nelement


"""auxiliary."""


def recover_device(data, device=None):
    if device is not None:
        return data.to(device)
    else:
        return data



def deepcopy_model(conf, model):
    # a dirty hack....
    tmp_model = deepcopy(model)
    if conf.track_model_aggregation:
        for tmp_para, para in zip(tmp_model.parameters(), model.parameters()):
            tmp_para.grad = para.grad.clone()
    return tmp_model


def get_model_difference(model1, model2):
    list_of_tensors = []
    for weight1, weight2 in zip(model1.parameters(),
                                model2.parameters()):
        tensor = get_diff_weights(weight1, weight2)
        list_of_tensors.append(tensor)
    return list_to_vec(list_of_tensors).norm().item()


def get_name_params_difference(named_parameters1, named_parameters2):
    """
        return named_parameters2 - named_parameters1
    """
    common_names = list(set(named_parameters1.keys()).intersection(set(named_parameters2.keys())))
    named_diff_parameters = {}
    for key in common_names:
        named_diff_parameters[key] = get_diff_weights(named_parameters1[key], named_parameters2[key])
    return named_diff_parameters


def get_diff_weights(weights1, weights2):
    """ Produce a direction from 'weights1' to 'weights2'."""
    if isinstance(weights1, list) and isinstance(weights2, list):
        return [w2 - w1 for (w1, w2) in zip(weights1, weights2)]
    elif isinstance(weights1, torch.Tensor) and isinstance(weights2, torch.Tensor):
        return weights2 - weights1
    else:
        raise NotImplementedError


def get_diff_states(states1, states2):
    """ Produce a direction from 'states1' to 'states2'."""
    return [
        v2 - v1
        for (k1, v1), (k2, v2) in zip(states1.items(), states2.items())
    ]


def list_to_vec(weights):
    """ Concatnate a numpy list of weights of all layers into one torch vector.
    """
    v = []
    direction = [d * np.float64(1.0) for d in weights]
    for w in direction:
        if isinstance(w, np.ndarray):
            w = torch.tensor(w)
        else:
            w = w.clone().detach()
        if w.dim() > 1:
            v.append(w.view(w.numel()))
        elif w.dim() == 1:
            v.append(w)
    return torch.cat(v)


def is_float(value):
    try:
        float(value)
        return True
    except:
        return False



"""gradient related"""
# TODO
def apply_gradient(param_groups, state, apply_grad_to_model=True):
    """
        SGD
    """
    for group in param_groups:
        weight_decay = group["weight_decay"]
        momentum = group["momentum"]
        dampening = group["dampening"]
        nesterov = group["nesterov"]

        for p in group["params"]:
            if p.grad is None:
                continue
            d_p = p.grad.data

            # get param_state
            param_state = state[p]

            # add weight decay.
            if weight_decay != 0:
                d_p.add_(p.data, alpha=weight_decay)

            # apply the momentum.
            if momentum != 0:
                if "momentum_buffer" not in param_state:
                    buf = param_state["momentum_buffer"] = torch.zeros_like(p.data)
                    buf.mul_(momentum).add_(d_p)
                else:
                    buf = param_state["momentum_buffer"]
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                if nesterov:
                    d_p = d_p.add(momentum, buf)
                else:
                    d_p = buf
            if apply_grad_to_model:
                p.data.add_(d_p, alpha=-group["lr"])
            else:
                p.grad.data = d_p




"""dataset related"""

def get_local_num_iterations(local_num, batch_size):
    return local_num // batch_size


def get_min_num_iterations(train_data_local_num_dict, batch_size):
    """
        This is used to get the minimum iteration of all clients.
        Note: For APSGD and SSPSGD, this function is different,
            because local client records their local epochs.
    """
    min_local_num = 10000000
    for worker_idx, local_num in train_data_local_num_dict.items():
        if min_local_num > local_num:
            min_local_num = local_num
    return min_local_num // batch_size


def get_max_num_iterations(train_data_local_num_dict, batch_size):
    """
        This is used to get the maximum iteration of all clients.
        Note: For APSGD and SSPSGD, this function is different,
            because local client records their local epochs.
    """
    max_local_num = 0
    for worker_idx, local_num in train_data_local_num_dict.items():
        if max_local_num < local_num:
            max_local_num = local_num
    return max_local_num // batch_size


def get_avg_num_iterations(train_data_local_num_dict, batch_size):
    """
        This is used to get the averaged iteration of all clients.
        Note: For APSGD and SSPSGD, this function is different,
            because local client records their local epochs.
    """
    sum_num = 0
    for worker_idx, local_num in train_data_local_num_dict.items():
        sum_num += local_num
    num_workers = len(train_data_local_num_dict.keys())
    return (sum_num // num_workers) // batch_size


def get_sum_num_iterations(train_data_local_num_dict, batch_size):
    """
        This is used to get the averaged iteration of all clients.
        Note: For APSGD and SSPSGD, this function is different,
            because local client records their local epochs.
    """
    sum_num = 0
    for worker_idx, local_num in train_data_local_num_dict.items():
        sum_num += local_num
    return sum_num // batch_size






""" data distribution """
def get_num_cls_in_batch(batch_data, cls_idx):
    return len(batch_data[batch_data == cls_idx])


def get_label_distribution(train_data_local_dict, class_num):
    local_cls_num_list_dict = {}
    for client in train_data_local_dict.keys():
        logging.info("In get_label_distribution: travelling client: {} ".format(client))
        local_cls_num_list_dict[client] = [0 for _ in range(class_num)]
        for _, labels in train_data_local_dict[client]:
            for cls_idx in range(class_num):
                local_cls_num_list_dict[client][cls_idx] += get_num_cls_in_batch(labels, cls_idx)
    return local_cls_num_list_dict




def get_selected_clients_label_distribution(local_cls_num_list_dict, class_num, client_indexes, min_limit=0):
    logging.info(local_cls_num_list_dict)
    selected_clients_label_distribution = [0 for _ in range(class_num)]
    for client_index in client_indexes:
        # selected_train_data_local_num_dict[client_index] = [0 for _ in range(class_num)]
        for cls_idx in range(class_num):
            selected_clients_label_distribution[cls_idx] += local_cls_num_list_dict[client_index][cls_idx]
    if min_limit > 0:
        for i in range(class_num):
            if selected_clients_label_distribution[i] < min_limit:
                selected_clients_label_distribution[i] = min_limit
    return selected_clients_label_distribution


def get_per_cls_weights(cls_num_list, beta=0.9999):
    effective_num = 1.0 - np.power(beta, cls_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    # per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)



""" cpu --- gpu """
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)










