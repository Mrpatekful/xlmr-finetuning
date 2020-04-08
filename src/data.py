"""
@author:    Patrik Purgai
@copyright: Copyright 2019, xlmr-finetuning
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.08.11.
"""

import os
import math
import json
import torch
import shutil
import functools
import itertools
import collections

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from tqdm import tqdm

from os.path import (
    join, dirname,
    exists, basename,
    splitext, abspath)


IGNORE_ID = -1


def group_elements(iterable, group_size):
    """
    Collect data into fixed-length chunks.
    """
    groups = [iter(iterable)] * group_size

    return itertools.zip_longest(*groups)


def int64_feature(value):
    """
    Creates an int64 feature from integers.
    """
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=value))


def read_file(file_name):
    """
    Reads lines from a file.
    """
    with open(file_name, 'r') as fh:
        for line in fh:
            yield line.strip()


def generate_examples(file_name):
    """
    Generates examples from the provided files.
    """
    for line in read_file(file_name):
        yield json.loads(line)


def transform_split(cfg, file_name, xlmr, label2id):
    """
    Transforms the provided split.
    """
    def is_not_none(e):
        """
        Helper function to filter nulls.
        """
        return e is not None

    def generate_groups():
        """
        Generates groups for serialization.
        """
        groups = group_elements(
            generate_examples(file_name),
            cfg.tfrecord_size)

        # pairing groups to unique numbers and 
        # filtering nulls from zip_longest
        groups = (
            list(filter(is_not_none, group))
            for group in groups
        )

        yield from groups

    split_name = splitext(basename(file_name))[0]

    tfrecord_name = join(
        cfg.data_dir,
        split_name + '.{}.tfrecord')

    tfrecord_name = abspath(tfrecord_name)

    encode_fn = functools.partial(
         encode_example,
         xlmr=xlmr,
         label2id=label2id)

    def generate_results():
        """
        Performs serialization and generates
        the resulting file names and sizes.
        """
        for idx, examples in enumerate(generate_groups()):
            # converting iterators to list so resources
            # are not shared in concurrent workers
            yield write_tfrecord(
                examples=examples,
                encode_fn=encode_fn,
                file_name=tfrecord_name.format(idx))

    # generates split sizes and filenames 
    # of the tfrecords
    tfrecord_paths, sizes = zip(*generate_results())

    return tfrecord_paths, sum(sizes)


def encode_example(tokens, labels, xlmr, label2id):
    """
    Converts the text examples to ids.
    """
    input_ids, label_ids = [], []

    merged = list(zip(tokens, labels))

    for token, label in merged:
        ids = xlmr.encode(token)[1:-1].tolist()
        input_ids.extend(ids)

        padding = [IGNORE_ID] * (len(ids) - 1)
        label_ids.extend([label2id[label]] + padding)

    input_ids = [xlmr.task.dictionary.bos()] + \
        input_ids + [xlmr.task.dictionary.eos()]

    label_ids = [IGNORE_ID] + label_ids + [IGNORE_ID]

    return input_ids, label_ids


def decode_example(input_ids, label_ids, xlmr, id2label):
    """
    Convert the id examples to text.
    """
    token_ids = []

    def decode_token(token):
        """
        Convenience function for converting a list
        of ids to text.
        """
        text = xlmr.decode(torch.tensor(token).long())
        return text.replace(' ', '')

    for token_id in input_ids:
        if token_id == xlmr.task.dictionary.pad(): break

        token_ids.append(token_id)

    # last item is the eos token so skipping it
    merged = list(zip(token_ids[:-1], label_ids))

    init_token, init_label = merged[1]
    init_label = id2label[init_label]

    tokens, labels, token = [], [init_label], [init_token]

    for sub_token_id, label_id in merged[2:]:
        if label_id != IGNORE_ID:
            tokens.append(decode_token(token))
            token.clear()

            labels.append(id2label[label_id])

        token.append(sub_token_id)

    if len(token) > 0:
        tokens.append(decode_token(token))

    # self checking the results
    assert len(tokens) == len(labels)

    return tokens, labels


def write_tfrecord(examples, encode_fn, file_name):
    """
    Converts the provided examples to ids and writes
    them to tfrecords.
    """
    def create_feature(example):
        """
        Creates a feature list from a document.
        """
        input_ids, label_ids = encode_fn(
            example['tokens'], example['labels'])

        features = {
            'input_ids': int64_feature(input_ids),
            'label_ids': int64_feature(label_ids)
        }

        return features

    with tf.io.TFRecordWriter(file_name) as writer:
        for example in examples:
            example = tf.train.Example(
                features=tf.train.Features(
                    feature=create_feature(example)))

            writer.write(example.SerializeToString())

    return file_name, len(examples)


def create_tfrecord_loader(
        cfg, tfrecord_paths, size, pad_id, shuffle=False):
    """
    Creates a generator that iterates through the
    dataset.
    """
    pad_id = tf.constant(pad_id, tf.int64)
    ignore_id = tf.constant(IGNORE_ID, tf.int64)

    def parse_example(example):
        """
        Parses a dialog from the serialized datafile.
        """
        features = {
            'input_ids': tf.io.VarLenFeature(tf.int64),
            'label_ids': tf.io.VarLenFeature(tf.int64)
        }

        parsed_example = \
            tf.io.parse_single_example(
                example, features=features)

        return {
            k: tf.sparse.to_dense(v) for k, v in
            parsed_example.items()
        }

    def compute_length(example):
        """
        Computes the length of the example.
        """
        return tf.size(example['input_ids'])

    def prepare_inputs(example):
        """
        Creates the attention mask tensor.
        """
        return example['input_ids'], example['label_ids']

    dataset = tf.data.TFRecordDataset(tfrecord_paths)

    if shuffle: dataset = dataset.shuffle(5000)

    dataset = dataset.map(
        parse_example,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)    

    dataset = dataset.padded_batch(
        batch_size=cfg.batch_size,
        padded_shapes={
            'input_ids': tf.TensorShape([None]),
            'label_ids': tf.TensorShape([None])
        },
        padding_values={
            'input_ids': pad_id,
            'label_ids': ignore_id
        })

    dataset = dataset.map(
        prepare_inputs,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.prefetch(
        tf.data.experimental.AUTOTUNE)

    return TensorflowDataset(dataset, size, cfg)


class TensorflowDataset:
    """
    Wraps the datset so it has a __len__ property.
    """

    def __init__(self, dataset, size, cfg):
        self.dataset = dataset
        self.size = math.ceil(size / cfg.batch_size)

    def __iter__(self):
        return self.dataset.as_numpy_iterator()

    def __len__(self):
        return self.size


def create_jsonl_loader(
        batch_size, data_path, encode_fn, pad_id):
    """
    Creates a loader that fetches examples directly
    from a jsonl file.
    """
    pad_id = tf.constant(pad_id, tf.int64)
    ignore_id = tf.constant(IGNORE_ID, tf.int64)

    def generate_converted():
        """
        Generator for converted examples.
        """
        for example in generate_examples(data_path):
            input_ids, label_ids = encode_fn(
                example['tokens'], example['labels'])

            yield {
                'input_ids': input_ids,
                'label_ids': label_ids
            }

    def prepare_inputs(example):
        """
        Creates the attention mask tensor.
        """
        return example['input_ids'], example['label_ids']

    dataset = tf.data.Dataset.from_generator(
        generate_converted,
        output_types={
            'input_ids': tf.int64,
            'label_ids': tf.int64
        },
        output_shapes={
            'input_ids': tf.TensorShape([None]),
            'label_ids': tf.TensorShape([None])
        })

    dataset = (
        dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes={
                'input_ids': tf.TensorShape([None]),
                'label_ids': tf.TensorShape([None])
            },
            padding_values={
                'input_ids': pad_id,
                'label_ids': ignore_id
            })
        .map(prepare_inputs)
    )

    return dataset


def generate_labels(cfg, split_files):
    """
    Generates labels from the train and valid splits.
    """
    for file_name in split_files:
        file_name = join(cfg.data_dir, file_name)

        for example in generate_examples(file_name):
            yield from example['labels']


def create_label2id(cfg):
    """
    If label2id doesn't exist then it is created.
    """
    label2id_model_path = join(
        cfg.model_dir, 'labels.json')

    # label2id is stored in the data dir and model dir
    if not exists(label2id_model_path):
        label2id_data_path = join(
            cfg.data_dir, 'labels.json')

        if not exists(label2id_data_path):
            label2id = {}

            for label in generate_labels(
                    cfg, ['train.jsonl', 'valid.jsonl']):
                if label not in label2id:
                    label2id[label] = len(label2id)

            with open(label2id_data_path, 'w') as fh:
                json.dump(label2id, fh)

        else:
            with open(label2id_data_path, 'r') as fh:
                label2id = json.load(fh)

        with open(label2id_model_path, 'w') as fh:
            json.dump(label2id, fh)

    with open(label2id_model_path, 'r') as fh:
        label2id = json.load(fh)

    return label2id


def create_dataset(cfg, xlmr, label2id):
    """
    Transforms the dataset and provides iterators to it.
    """
    assert exists(cfg.data_dir), \
        '{} does not exist.'.format(cfg.data_dir)

    metadata_path = join(
        cfg.data_dir, 'metadata.json')

    if not exists(metadata_path):
        # if dataset does not exist then create it
        # by tokenizing the raw files

        splits = [
            join(cfg.data_dir, 'train.jsonl'),
            join(cfg.data_dir, 'valid.jsonl')
        ]

        splits = tqdm(
            splits, 
            desc='Converting to tfrecord',
            leave=False
        )

        train, valid = [
            transform_split(
                cfg=cfg,
                file_name=file_name,
                xlmr=xlmr,
                label2id=label2id)
            for file_name in splits]

        train_tfrecords, train_size = train
        valid_tfrecords, valid_size = valid

        print('Saving metadata to {}'.format(
            metadata_path))

        # save the location of the files in a metadata
        # json object and delete the file in case of
        # interrupt so it wont be left in corrupted state
        with open(metadata_path, 'w') as fh:
            try:
                json.dump({
                    'train': train,
                    'valid': valid
                }, fh)
            except KeyboardInterrupt:
                shutil.rmtree(metadata_path)

    print('Loading metadata from {}'.format(
        metadata_path))

    with open(metadata_path, 'r') as fh:
        metadata = json.load(fh)

    train_tfrecords, train_size = metadata['train']
    valid_tfrecords, valid_size = metadata['valid']

    pad_id = xlmr.task.dictionary.pad()

    train_dataset = create_tfrecord_loader(
        cfg=cfg,
        tfrecord_paths=train_tfrecords,
        size=train_size,
        pad_id=pad_id,
        shuffle=True)

    valid_dataset = create_tfrecord_loader(
        cfg=cfg,
        tfrecord_paths=valid_tfrecords,
        size=valid_size,
        pad_id=pad_id)

    return train_dataset, valid_dataset

