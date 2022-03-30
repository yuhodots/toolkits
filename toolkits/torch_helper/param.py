"""
Parameter manipulation tools.
"""

import math
import numpy as np
import torch


def freeze_selected_param(model, target):
    """ Freeze layers which is in 'target'
    notice) This feature has not yet been tested.
    :param model: pytorch model
    :param target: layer name list
    :return: processed model
    """
    for param in model.parameters():
        param.requires_grad = True

    for name, param in model.named_parameters():
        if name in target:
            param.requires_grad = False

    return model


def get_important_param_idx(model, ratio, inverse=False):
    """ Get indices of parameters which has large absolute value
    :param model: pytorch model
    :param ratio: ratio of important parameters
    :param inverse: if inverse=True, select unimportant weights
    :return: dictionary of the indices of selected parameters
    """
    param_dict = dict(model.named_parameters())

    # Remove not trainable parameter from parameter dictionary
    remove_list = [k for k, v in param_dict.items() if not v.requires_grad]
    for k in remove_list:
        param_dict.pop(k)

    # Pre-processing parameter data
    param_flatten_dict = {}
    param_len_dict = {}
    for k, v in param_dict.items():
        param_dict[k] = param_dict[k].data
        param_flatten_dict[k] = torch.reshape(param_dict[k].detach().data, [-1])
        param_len_dict[k] = param_flatten_dict[k].shape[0]

    # Select the important weight (large absolute value)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_st = math.floor(num_trainable_params * ratio)  # The number of session trainable parameters
    all_weight = np.abs(np.array(sum([list(v.cpu().numpy()) for v in param_flatten_dict.values()], [])))
    if inverse:
        sel_weight_idx = all_weight.argsort()[:num_st]
    else:
        sel_weight_idx = all_weight.argsort()[::-1][:num_st]

    # Get indices of selected parameters
    count = 0
    param_idx_dict = {}
    for k, v in param_len_dict.items():
        param_flattened_idx_list = [w_idx - count for w_idx in sel_weight_idx
                                    if (w_idx >= count) & (w_idx < count + v)]
        param_idx_dict[k] = [np.unravel_index(flattened_idx, param_dict[k].shape)
                             for flattened_idx in param_flattened_idx_list]
        count += v

    return param_idx_dict

