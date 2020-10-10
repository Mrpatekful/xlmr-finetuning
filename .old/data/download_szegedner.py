"""
@author:    Patrik Purgai
@copyright: Copyright 2019, xlmr-hungarian
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.04.04.
"""

import os
import shutil
import math
import argparse
import zipfile
import json
import collections
import requests
import itertools

from tempfile import TemporaryFile
from os.path import join, dirname, abspath, exists


URL = "https://rgai.sed.hu/sites/rgai.sed.hu/files/"

FILE_NAME = "business_NER"

DATA_DIR = join(abspath(dirname(__file__)))


def download(dump_path):
    """
    Downloads the the business NER data file.
    """
    request = requests.get(URL + FILE_NAME + ".zip", stream=True)

    with open(dump_path, "wb") as fh:
        for chunk in request.iter_content(chunk_size=1000):
            fh.write(chunk)


def group_elements(iterable, group_size):
    """
    Collect data into fixed-length chunks.
    """
    groups = [iter(iterable)] * group_size

    return itertools.zip_longest(*groups)


def read_file(file_name):
    """
    Reads the lines of the provided file.
    """
    with open(file_name, "r", encoding="latin-1") as fh:
        for line in fh:
            yield line.strip()


def generate_sequences(file_name):
    """
    Generates sentences with tags from a file.
    """
    sequence = []

    for line in read_file(file_name):
        split = line.split("\t")
        if len(split) > 1:
            # using only the token and ner tag
            # which is the final element in the row
            label = split[-1]
            label = "O" if label == "0" else label
            sequence.append((split[0], label))
        elif len(sequence) > 0:
            yield list(zip(*sequence))
            sequence = []


def generate_jsonl(file_name):
    """
    Converts conll formated file to jsonl.
    """
    for tokens, labels in generate_sequences(file_name):
        yield {"tokens": tokens, "labels": correct_labels(labels)}


def correct_labels(labels):
    """
    Converts the begining of a I-x label seq to B-x.
    """

    def correct_label(args):
        idx, curr = args

        prev = "" if idx - 1 < 0 else labels[idx - 1]

        prev_splits = prev.split("-")
        curr_splits = curr.split("-")

        if len(curr_splits) > 1:
            curr_prefix, curr_postfix = curr_splits

            if len(prev_splits) > 1:
                prev_prefix, prev_postfix = prev_splits

                if curr_postfix != prev_postfix:
                    return curr.replace("I-", "B-")

                else:
                    return curr

            else:
                return curr.replace("I-", "B-")

        return curr

    return [label for label in map(correct_label, enumerate(labels))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=join(DATA_DIR, "szegedner"),
        help="Path of the data directory.",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Download dataset even if it exists.",
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        default=9,
        help="Number of folds for the cross validation.",
    )
    parser.add_argument(
        "--min_class_example",
        type=int,
        default=10,
        help="Minimum number of example in a fold for a class.",
    )
    parser.add_argument(
        "--valid_idx", type=int, default=1, help="Index of the validation split."
    )

    args = parser.parse_args()

    assert args.valid_idx < args.num_folds

    os.makedirs(args.data_dir, exist_ok=True)

    extract_path = join(args.data_dir, FILE_NAME)
    download_path = extract_path + ".zip"

    if not exists(extract_path) or args.force_download:
        if exists(extract_path):
            shutil.rmtree(extract_path)

        if not exists(download_path) or args.force_download:
            download(download_path)

        with zipfile.ZipFile(download_path) as zf:
            zf.extractall(path=extract_path)

    raw_data_path = join(args.data_dir, FILE_NAME, "hun_ner_corpus.txt")

    size = 0
    train_path = join(args.data_dir, "train.jsonl")
    valid_path = join(args.data_dir, "valid.jsonl")

    def write_fold(fold, file_handle, labels):
        """
        Writes the given fold the the file.
        """
        for example in fold:
            labels.update(json.loads(example)["labels"])
            file_handle.write(example)

    def is_not_none(value):
        """
        Returns value is not None.
        """
        return value is not None

    train_labels = collections.Counter()
    valid_labels = collections.Counter()

    with TemporaryFile("w+", dir=args.data_dir) as fh:
        for json_example in generate_jsonl(raw_data_path):
            fh.write(json.dumps(json_example) + "\n")
            size += 1

        fh.flush()
        fh.seek(0)

        # computing the fold size note that the last
        # fold might be slightly smaller
        fold_size = math.ceil(size / args.num_folds)
        folds = (filter(is_not_none, g) for g in group_elements(fh, fold_size))

        with open(train_path, "w") as ft:
            for idx, fold in enumerate(folds):
                if idx == args.valid_idx:
                    with open(valid_path, "w") as fv:
                        write_fold(fold, fv, valid_labels)

                else:
                    write_fold(fold, ft, train_labels)

    labels = {*train_labels, *valid_labels}

    assert min(train_labels.get(l, 0) for l in labels) > args.min_class_example

    assert min(valid_labels.get(l, 0) for l in labels) > args.min_class_example


if __name__ == "__main__":
    main()
