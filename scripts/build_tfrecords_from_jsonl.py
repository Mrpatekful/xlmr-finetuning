"""
@author:    Patrik Purgai
@copyright: Copyright 2020, dialogue-generation
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2020.05.30.
"""

import os
import argparse
import glob
import json
import transformers
import multiprocessing

import tensorflow as tf


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def create_feature_list(document, tokenizer):
    feature_list = tf.train.FeatureList(
        feature=[int64_feature(tokenizer.encode(example)) for example in document]
    )

    return feature_list


def create_feature(example):
    input_ids, label_ids = encode_fn(example['tokens'], example['labels'])

    features = {
        'input_ids': int64_feature(input_ids),
        'label_ids': int64_feature(label_ids)
    }

    return features


def encode_example(tokens, labels, tokenizer, label2id, no_label):
    input_ids, label_ids = [], []

    merged = list(zip(tokens, labels))

    for token, label in merged:
        ids = tokenizer.encode(token)[1:-1]
        input_ids.extend(ids)

        padding = [no_label] * (len(ids) - 1)
        label_ids.extend([label2id[label]] + padding)

    input_ids.insert(0, tokenizer.convert_tokens_to_ids(tokenizer.bos_token))
    input_ids.append(tokenizer.convert_tokens_to_ids(tokenizer.eos_token))

    label_ids.insert(0, no_label)
    label_ids.append(no_label)

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



def read_jsonl(file_name):
    with open(file_name) as fh:
        for line in fh:
            yield json.loads(line)


def write_examples(job_id, args, tokenizer):
    file_names = sorted(glob.glob(args.pattern))
    file_names = [
        file_name
        for (idx, file_name) in enumerate(file_names)
        if idx % args.n_workers == job_id
    ]

    for file_no, file_name in enumerate(file_names):
        dialogues = read_dialogues(file_name)
        tfrecord_name = os.path.join(args.output_dir, f"{file_no}.tfrecord")
        write_tfrecord(tfrecord_name, tokenizer, dialogues)


def write_tfrecord(file_name, tokenizer, dialogues):
    with tf.io.TFRecordWriter(file_name) as writer:
        for dialogue in dialogues:
            feature_list = {"dialogue": create_feature_list(dialogue, tokenizer)}

            example = tf.train.Example(
                features=tf.train.Features(
                    feature=create_feature(example)))

            writer.write(example.SerializeToString())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pattern",
        required=True,
        type=str,
        help="Pattern of the input files for a single split.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Directory of the output tfrecords.",
    )
    parser.add_argument(
        "--n_workers",
        default=1,
        type=int,
        help="Number of concurrent workers for building.",
    )
    parser.add_argument(
        "--pretrained",
        required=True,
        type=str,
        help="Name of the pretrained auto config for the tokenizer.",
    )
    args = parser.parse_args()

    tokenizer = transformers.XLMRobertaTokenizer.from_pretrained(args.pretrained)
    print(tokenizer.convert_tokens_to_ids((tokenizer.bos_token, tokenizer.eos_token)))

    return

    os.makedirs(args.output_dir, exist_ok=True)

    n_input_files = len(glob.glob(args.pattern))
    n_output_files = len(glob.glob(os.path.join(args.output_dir, "*.tfrecord")))

    if n_input_files == n_output_files:
        return



    if args.n_workers > 1:
        write_examples(0, args)

    else:
        jobs = []
        for idx in range(args.n_workers):
            job = multiprocessing.Process(
                target=write_examples, args=(idx, args, tokenizer)
            )
            jobs.append(job)
            job.start()

        for job in jobs:
            job.join()


if __name__ == "__main__":
    main()
