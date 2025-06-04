import os.path
import pathlib

import torch
import logging
import numpy as np
import pickle as pkl
import json

from pathlib import Path

from torch.utils.data import DataLoader, TensorDataset



def dump_pkl(data, fname):
    if type(fname) == Path:
        fname = str(fname)
    with open(fname, 'wb') as f:
        pkl.dump(data, f)


def load_pkl(fname, mode='pkl'):
    if type(fname) == Path or type(fname) == pathlib.PosixPath:
        fname = str(fname)
    if fname.endswith('.pkl'):
        with open(fname, 'rb') as f:
            data = pkl.load(f)
    elif fname.endswith('.json'):
        with open(fname, "r") as f:
            data = json.load(f)
    else:
        raise ValueError(f"load_pkl: invalid file type '{fname.split('.')[-1]}', must be pkl or json")
    return data


def _get_outputs(inference_fn, data, model, device, batch_size=512):

    _data = DataLoader(TensorDataset(data), shuffle=False, batch_size=batch_size)

    try:
        _y_out = []
        for x in _data:
            _y = inference_fn(x[0].to(device))
            _y_out.append(_y.cpu())
        return torch.vstack(_y_out)
    except RuntimeError as re:
        if "CUDA out of memory" in str(re):
            model.to('cpu')
            outputs = _get_outputs(inference_fn, data, model, 'cpu')
            model.to('cuda')
            return outputs
        else:
            raise re


def _get_targets(inference_fn, data, model, device):
    return torch.argmax(_get_outputs(inference_fn, data, model, device), dim=1)

