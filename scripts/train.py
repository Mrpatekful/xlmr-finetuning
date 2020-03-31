"""
@author:    Patrik Purgai
@copyright: Copyright 2019, xlmr-finetuning
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.09.30.
"""

import sys
import os
import ignite
import functools
import collections
import argparse
import random
import torch
import json
import hydra

import numpy as np
import tensorflow as tf

from tabulate import tabulate
from datetime import datetime

from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar
from ignite.metrics import RunningAverage
from ignite.handlers import ModelCheckpoint

from os.path import (
    join, dirname,
    basename, exists,
    abspath, isdir)

PROJECT_PATH = join(abspath(dirname(__file__)), '..')
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

from src.model import (
    create_model,
    create_pretrained)

from src.data import (
    create_dataset,
    create_label2id)


def set_random_seed(cfg):
    """
    Sets the random seed for training.
    """
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)

    if cfg.cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_optimizer(cfg, model):
    """
    Creates an adam optimizer with correct weight
    decay method.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        eps=cfg.adam_epsilon,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay)

    return optimizer


def create_scheduler(cfg, optimizer, train_size):
    """
    Creates a scheduler with warmup and linear decay.
    """
    total_steps = train_size // \
        cfg.grad_accum_steps * cfg.max_epochs

    warmup_steps = total_steps * cfg.warmup_prop

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)

        return max(0, total_steps - current_step) / \
            max(1, total_steps - warmup_steps)

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda)


@hydra.main(config_path='config.yaml', strict=False)
def main(cfg):
    """
    Performs training, validation and testing.
    """
    assert isdir(cfg.data_dir), \
        '`data_dir` must be a valid path.'

    cfg.cuda = torch.cuda.is_available() \
        and not cfg.no_cuda

    cfg.model_dir = os.getcwd()
    
    # setting random seed for reproducibility
    if cfg.seed: set_random_seed(cfg)

    device = torch.device(
        'cuda' if cfg.cuda else 'cpu')

    os.makedirs(cfg.model_dir, exist_ok=True)

    label2id = create_label2id(cfg)
    cfg.num_labels = len(label2id)

    xlmr = create_pretrained(
        cfg.model_type, cfg.force_download)

    # creating dataset split loaders
    datasets = create_dataset(cfg, xlmr, label2id)

    train_dataset, valid_dataset = datasets

    def compute_loss(batch):
        """
        Computes the forward pass and returns the
        cross entropy loss.
        """
        inputs, labels = [
            torch.from_numpy(tensor).to(device).long() 
            for tensor in batch
        ]

        logits = model(inputs)

        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)

        loss = torch.nn.functional.cross_entropy(
            logits, labels, ignore_index=-1)

        return loss

    def train_step(engine, batch):
        """
        Propagates the inputs forward and updates
        the parameters.
        """
        step = engine.state.iteration

        model.train()

        loss = compute_loss(batch)

        backward(loss)

        if cfg.clip_grad_norm is not None:
            clip_grad_norm(cfg.clip_grad_norm)

        if step % cfg.grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # restoring the averaged loss across steps
        loss *= cfg.grad_accum_steps

        return loss.item()

    def eval_step(engine, batch):
        """
        Propagates the inputs forward without
        storing any gradients.
        """
        model.eval()

        with torch.no_grad():
            loss = compute_loss(batch)

        return loss.item()

    def backward(loss):
        """
        Backpropagates the loss in either mixed or
        normal precision mode.
        """
        if cfg.fp16:
            with amp.scale_loss(loss, optimizer) as sc:
                sc.backward()

        else: loss.backward()

    def clip_grad_norm(max_norm):
        """
        Applies gradient clipping.
        """
        if cfg.fp16:
            params = amp.master_params(optimizer)
        else:
            params = model.parameters()

        torch.nn.utils.clip_grad_norm_(params, max_norm)

    trainer = Engine(train_step)
    validator = Engine(eval_step)

    checkpoint = ModelCheckpoint(
        cfg.model_dir,
        cfg.model_type,
        n_saved=5,
        save_as_state_dict=True,
        score_function=lambda e: -e.state.metrics['loss'])

    last_ckpt_path = cfg.ckpt_path

    if last_ckpt_path is not None:
        msg = 'Loading state from {}'
        print(msg.format(basename(last_ckpt_path)))

        last_state = torch.load(
            last_ckpt_path, map_location=device)

    model = create_model(xlmr, len(label2id), cfg)
    model = model.to(device)

    del xlmr.model

    optimizer = create_optimizer(cfg, model)

    scheduler = create_scheduler(
        cfg, optimizer, len(train_dataset))

    # using apex if required and loading its state
    if cfg.fp16:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O2')

        if last_ckpt_path is not None and \
                'amp' in last_state:
            amp.load_state_dict(last_state['amp'])

    if last_ckpt_path is not None:
        model.load_state_dict(last_state['model'])
        optimizer.load_state_dict(last_state['optimizer'])
        scheduler.load_state_dict(last_state['scheduler'])

    checkpoint_dict = {
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler
    }

    if cfg.fp16: checkpoint_dict['amp'] = amp

    validator.add_event_handler(
        Events.COMPLETED,
        checkpoint,
        checkpoint_dict)

    metric = RunningAverage(output_transform=lambda x: x)
    metric.attach(trainer, 'loss')
    metric.attach(validator, 'loss')

    pbar = ProgressBar()
    pbar.attach(trainer, metric_names=['loss'])

    history_path = join(cfg.model_dir, 'history.json')
    history = collections.defaultdict(list)
    headers = ['epoch', 'train_loss', 'valid_loss']

    if exists(history_path):
        with open(history_path, 'r') as fh:
            history = json.load(fh)

    def record_history(results):
        """
        Records the results to the history.
        """
        for header in headers:
            history[header].append(results[header])

        with open(history_path, 'w') as fh:
            json.dump(history, fh)

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_results(engine):
        """
        Logs the training results.
        """
        validator.run(valid_dataset)

        record_history({
            'epoch': engine.state.epoch,
            'train_loss': engine.state.metrics['loss'],
            'valid_loss': validator.state.metrics['loss']
        })

        data = list(zip(*[history[h] for h in headers]))
        table = tabulate(data, headers, floatfmt='.3f')

        print(table.split('\n')[-1])

    data = list(zip(*[history[h] for h in headers]))

    print()
    print(cfg.pretty())

    print()
    print('***** Running training *****')

    print()
    print(tabulate(data, headers, floatfmt='.3f'))

    trainer.run(train_dataset, cfg.max_epochs)


if __name__ == '__main__':
    main()

