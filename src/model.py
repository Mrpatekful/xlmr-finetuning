"""
@author:    Patrik Purgai
@copyright: Copyright 2019, xlmr-finetuning
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.09.30.
"""

import os
import re
import torch
import tarfile
import requests

import numpy as np

from tqdm import tqdm

from fairseq.models.roberta import XLMRModel
from fairseq.utils import get_activation_fn

from os.path import (
    join, dirname,
    abspath, exists)


def create_model(xlmr, num_labels, cfg):
    """
    Creates the pretrained initalized model.
    """
    if cfg.fp16:
        # using apex fused layernorm during training
        # for better fp16 compatibility
        from apex.normalization import FusedLayerNorm \
            as LayerNorm

    else:
        from torch.nn.modules import LayerNorm

    args = xlmr.model.args

    args.layer_drop_p = cfg.layer_drop_p
    args.vocab_size = len(xlmr.task.dictionary)
    args.padding_idx = xlmr.task.dictionary.pad()

    encoder = Encoder(args, LayerNorm)

    # initializing the encoder part of the classifier
    # from the pretrained xlmr model
    load_weights(encoder, xlmr)

    output = torch.nn.Linear(
        encoder.args.encoder_embed_dim,
        num_labels)

    output.weight.data.normal_(mean=0.0, std=0.02)
    output.bias.data.zero_()

    classifier = torch.nn.Sequential(
        encoder, torch.nn.Dropout(p=0.1), output)

    return classifier


def create_pretrained(model_type, force_download=False):
    """
    Downloads and creates the pretrained assets.
    """
    project_dir = join(abspath(dirname(__file__)), '..')
    cache_dir = join(project_dir, '.cache')

    os.makedirs(cache_dir, exist_ok=True)

    model_dir = join(cache_dir, model_type)

    if not exists(model_dir) or force_download:
        download(model_type, cache_dir)

    xlmr = XLMRModel.from_pretrained(
        model_dir, checkpoint_file='model.pt')

    return xlmr


def download(model_type, cache_dir):
    """
    Downloads the provided model from 
    args.model_type.
    """
    url = 'https://dl.fbaipublicfiles.com/fairseq/models/{}.tar.gz'

    # determining the dump path for pretrained models
    download_path = join(cache_dir, 'xlmr.tar.gz')

    request = requests.get(
        url.format(model_type), stream=True)

    with open(download_path, 'wb') as fh:
        file_size = request.headers['content-length']
        file_size = int(file_size)

        with tqdm(total=file_size, leave=False) as pbar:
            for chunk in request.iter_content(1000):
                fh.write(chunk)
                pbar.update(1000)

    with tarfile.open(download_path, 'r:gz') as tf:
        tf.extractall(cache_dir)

    os.remove(download_path)


class PositionEmbedding(torch.nn.Embedding):
    """
    Position embedding for xlmr model.
    """
    def __init__(
            self, num_embeddings, embedding_dim,
            padding_idx):
        # if padding_idx is specified then offset the 
        # embedding ids by this index and adjust
        # num_embeddings appropriately
        if padding_idx is not None:
            num_embeddings = num_embeddings + \
                padding_idx + 1

        super().__init__(
            num_embeddings, embedding_dim,
            padding_idx)

        self.padding_idx = padding_idx

        self.max_positions = \
            num_embeddings - padding_idx - 1

    def forward(self, inputs):
        mask = inputs.ne(self.padding_idx).long()

        # onnx export is not working for some reason
        # with the original cumsum operation
        positions = torch.arange(mask.size(1)).long()
        positions = positions.to(inputs.device) * \
            mask + self.padding_idx + 1

        return super().forward(positions)


class Encoder(torch.nn.Module):
    """
    Implements the transformer encoder.
    """

    def __init__(self, args, layer_norm):
        super().__init__()

        self.args = args

        self.embed_tokens = torch.nn.Embedding(
            args.vocab_size,
            args.encoder_embed_dim,
            self.args.padding_idx)

        self.embed_positions = PositionEmbedding(
            args.max_positions,
            args.encoder_embed_dim,
            self.args.padding_idx)

        self.layers = torch.nn.ModuleList([
            EncoderLayer(args, layer_norm)
            for _ in range(args.encoder_layers)
        ])

        self.emb_layer_norm = layer_norm(
            args.encoder_embed_dim)

    def forward(self, inputs):
        attn_mask = inputs.eq(self.args.padding_idx)

        embeds = self.embed_tokens(inputs)
        embeds += self.embed_positions(inputs)
        embeds = self.emb_layer_norm(embeds)

        embeds = torch.nn.functional.dropout(
            embeds, p=0.1, training=self.training)

        embeds *= \
            1 - attn_mask.unsqueeze(2).type_as(embeds)

        outputs = embeds.transpose(0, 1)

        for layer in self.layers:
            outputs, _ = layer(outputs, attn_mask)

        outputs = outputs.transpose(0, 1)

        return outputs


class EncoderLayer(torch.nn.Module):
    """
    Implements the transformer encoder layer.
    """

    def __init__(self, args, layer_norm):
        super().__init__()

        self.self_attn = torch.nn.MultiheadAttention(
            args.encoder_embed_dim,
            args.encoder_attention_heads,
            dropout=0.1)

        self.activation_fn = get_activation_fn(
            args.activation_fn)

        # layer norm associated with the self
        # attention layer
        self.self_attn_layer_norm = layer_norm(
            args.encoder_embed_dim)

        self.fc1 = torch.nn.Linear(
            args.encoder_embed_dim,
            args.encoder_ffn_embed_dim)

        self.fc2 = torch.nn.Linear(
            args.encoder_ffn_embed_dim,
            args.encoder_embed_dim)

        # layer norm associated with the 
        # position wise feed-forward NN
        self.final_layer_norm = layer_norm(
            args.encoder_embed_dim)

    def forward(self, inputs, attn_mask):
        residual = inputs

        inputs, attn_weights = self.self_attn(
            query=inputs,
            key=inputs,
            value=inputs,
            key_padding_mask=attn_mask,
            need_weights=False)

        inputs = torch.nn.functional.dropout(
            inputs, p=0.1, training=self.training)

        inputs = residual + inputs
        inputs = self.self_attn_layer_norm(inputs)

        residual = inputs
        inputs = self.activation_fn(self.fc1(inputs))
        inputs = torch.nn.functional.dropout(
            inputs, p=0.1, training=self.training)

        inputs = self.fc2(inputs)
        inputs = torch.nn.functional.dropout(
            inputs, p=0.1, training=self.training)

        inputs = residual + inputs
        inputs = self.final_layer_norm(inputs)

        return inputs, attn_weights


def load_weights(model, xlmr):
    """
    Loads the weights of the provided model.
    """
    pretrained_state = \
        xlmr.model.decoder.sentence_encoder.state_dict()

    loaded_state = {}

    for key in model.state_dict():
        if 'in_proj' in key:
            # torch multihead attention implementation
            # stores the qkv parameters as a single
            # parameter
            from_key = re.sub(
                r'in_proj_(.*)', r'{}_proj.\1', key)
            
            loaded_state[key] = torch.cat([
                pretrained_state[from_key.format(s)]
                for s in ['q', 'k', 'v']
            ], dim=0)

        else:
            loaded_state[key] = pretrained_state[key]

    model.load_state_dict(loaded_state)


if __name__ == '__main__':
    xlmr = create_pretrained('xlmr.base')

    args = xlmr.model.args
    args.padding_idx = xlmr.task.dictionary.pad()
    args.vocab_size = len(xlmr.task.dictionary)

    encoder = Encoder(args, torch.nn.LayerNorm)

    load_weights(encoder, xlmr)

    encoder.eval()
    xlmr.model.eval()

    sample_input = xlmr.encode('Ez egy teszt')
    sample_input = sample_input.long().unsqueeze(0)

    def to_numpy(tensor):
        """
        Casting a tensor to numpy.
        """
        return tensor.cpu().numpy()

    with torch.no_grad():
        np.testing.assert_allclose(
            to_numpy(encoder(sample_input)),
            to_numpy(xlmr.model(sample_input, True)[0]),
            atol=1e-05)

