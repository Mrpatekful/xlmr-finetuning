"""
@author:    Patrik Purgai
@copyright: Copyright 2019, xlmr-finetuning
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.04.04.
"""

import os
import sys
import torch
import functools
import onnx
import json
import argparse
import hydra

import numpy as np

from os.path import (
    join, dirname,
    abspath)

PROJECT_DIR = join(abspath(dirname(__file__)), '..')
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

from src.data import encode_example

from src.model import (
    create_model,
    create_pretrained)


@hydra.main(config_path='config.yaml', strict=False)
def main(cfg):
    """
    Converts the model to onnx format.
    """
    cfg.cuda = not cfg.no_cuda and \
        torch.cuda.is_available()

    model_dir = abspath(dirname(cfg.ckpt_path))
    output_dir = os.getcwd()

    device = torch.device(
        'cuda' if cfg.cuda else 'cpu')

    os.makedirs(output_dir, exist_ok=True)

    labels_path = join(model_dir, 'labels.json') \
        if cfg.labels_path is None else \
        cfg.labels_path

    with open(labels_path, 'r') as fh:
        label2id = json.load(fh)

    xlmr = create_pretrained(
        cfg.model_type, cfg.force_download)

    encode_fn = functools.partial(
        encode_example,
        xlmr=xlmr,
        label2id=label2id)

    model = create_model(xlmr, len(label2id), cfg)
    model.to(device)

    state_dict = torch.load(
        cfg.ckpt_path, map_location=device)

    model.load_state_dict(state_dict['model'])
    model.eval()

    sample_input = xlmr.encode('Ez egy teszt')
    sample_input = sample_input[None, :].to(device)

    output_path = join(
        output_dir, cfg.model_type + '.onnx')

    torch.onnx.export(
        model,
        sample_input,
        output_path,
        export_params=True,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0 : 'batch_size', 1: 'sequence'},
            'output': {0 : 'batch_size', 1: 'sequence'}
        },
        verbose=True)

    print()
    print('***** Export *****')
    print()

    print('Model exported to {}.'.format(output_dir))
    print()

    onnx_model = onnx.load(output_path)
    # only works with onnx 1.5 for some reason
    # 1.6 produces segmentation fault error
    onnx.checker.check_model(onnx_model)


if __name__ == '__main__':
    main()

