"""
@author:    Patrik Purgai
@copyright: Copyright 2019, xlmr-hungarian
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.07.20.
"""

import sys
import os
import torch
import subprocess
import argparse
import functools
import random
import json
import hydra
import yaml

import numpy as np
import tensorflow as tf

from tqdm import tqdm

from os.path import (
    join, dirname,
    abspath, exists)

PROJECT_PATH = join(abspath(dirname(__file__)), '..')
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

from src.model import (
    create_model,
    create_pretrained)

from src.data import (
    encode_example,
    decode_example,
    create_jsonl_loader)


def try_load_config(model_dir):
    """
    Tries to load the training config from the model.
    """
    config_dir = join(model_dir, '.hydra')

    if exists(config_dir):
        try:
            with open(join(config_dir, 'config.yaml')) as fh:
                return yaml.safe_load(fh)
        except Exception:
            pass


@hydra.main(config_path='config.yaml', strict=False)
def main(cfg):
    """
    Performs evaluation.
    """
    cfg.cuda = torch.cuda.is_available()

    assert cfg.ckpt_path is not None, \
        'ckpt_path must be given'

    model_dir = abspath(dirname(cfg.ckpt_path))
    output_dir = os.getcwd()

    device = torch.device(
        'cuda' if cfg.cuda else 'cpu')

    output_path = join(output_dir, 'results.ner')

    labels_path = join(model_dir, 'labels.json') \
        if cfg.labels_path is None else \
        cfg.labels_path

    with open(labels_path, 'r') as fh:
        label2id = json.load(fh)

    id2label = {v: k for k, v in label2id.items()}

    xlmr = create_pretrained(
        cfg.model_type, cfg.force_download)

    encode_fn = functools.partial(
        encode_example,
        xlmr=xlmr,
        label2id=label2id)

    decode_fn = functools.partial(
        decode_example,
        xlmr=xlmr,
        id2label=id2label)

    model = create_model(xlmr, len(label2id), cfg)
    model = model.to(device)

    state_dict = torch.load(
        cfg.ckpt_path, map_location=device)

    model.load_state_dict(state_dict['model'])
    model.eval()

    def to_list(tensor):
        """
        Converts the provided tensor to a python list.
        """
        return tensor.cpu().numpy().tolist()

    def to_torch(tensor):
        """
        Converts the provided tf array to torch
        tensor.
        """
        return torch.from_numpy(tensor.numpy()).to(device)

    pad_id = xlmr.task.dictionary.pad()
    dataset = create_jsonl_loader(
        cfg.batch_size, cfg.eval_data_path,
        encode_fn, pad_id)

    print()
    print('***** Running evaluation *****')
    print()

    results = []
    with torch.no_grad():
        for batch in tqdm(dataset, leave=False):
            input_ids, label_ids = batch
    
            input_ids = to_torch(input_ids).long()
            label_ids = to_torch(label_ids).long()
    
            logits = model(input_ids)
    
            pred_ids = logits.argmax(dim=-1)
    
            lists = zip(
                to_list(pred_ids),
                to_list(label_ids),
                to_list(input_ids)
            )
    
            for pred_list, label_list, token_list in lists:
                pred_list = [
                    (pred if label != -1 else -1)
                    for pred, label in
                    zip(pred_list, label_list)
                ]

                tokens, labels = decode_fn({
                    'token_ids': token_list,
                    'label_ids': label_list
                })
    
                _, preds = decode_fn({
                    'token_ids': token_list,
                    'label_ids': pred_list
                })
    
                results.append((tokens, labels, preds))

    outputs = []
    for result in results:
        outputs.append('\n'.join(
            '{} {} {}'.format(*values)
            for values in zip(*result)
        ))

    with open(output_path, 'w') as fh:
        fh.write('\n\n'.join(outputs))

    command = '{} < {}'.format(
        join(PROJECT_PATH, 'scripts', 'conlleval'),
        output_path)

    result = subprocess.check_output(
        command,
        shell=True,
        stderr=subprocess.STDOUT)

    result = result.decode('utf-8')

    print(result)

    stats_path = join(output_dir, 'results.txt')
    with open(stats_path, 'w') as fh:
        config_str = try_load_config(model_dir)
        if config_str is not None:
            print(yaml.dump(config_str), file=fh)
        print(result, file=fh)


if __name__ == '__main__':
    main()

