"""
@author:    Patrik Purgai
@copyright: Copyright 2019, xlmr-finetuning
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.04.04.
"""

import os
import shutil
import math
import argparse
import zipfile
import random
import tqdm
import json
import collections
import requests
import tempfile
import itertools

URL = "https://rgai.sed.hu/sites/rgai.sed.hu/files/"
FILE_NAME = "business_NER"

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))


def download(file_name, url):
    with requests.Session() as session:
        response = session.get(url, stream=True, timeout=5)
        loop = tqdm.tqdm(desc="Downloading", unit="B", unit_scale=True)

        with open(file_name, "wb") as fh:
            for chunk in response.iter_content(1024):
                if chunk:
                    loop.update(len(chunk))
                    fh.write(chunk)


def group_elements(iterable, group_size):
    groups = [iter(iterable)] * group_size

    return itertools.zip_longest(*groups)


def read_file(file_name):
    with open(file_name, "r", encoding="latin-1") as fh:
        for line in fh:
            yield line.strip()


def generate_sequences(file_name):
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
    for tokens, labels in generate_sequences(file_name):
        yield {"tokens": tokens, "labels": fix_labels(labels)}


def fix_labels(labels):
    def fix_label(idx, curr):
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

    return [fix_label(i, l) for i, l in enumerate(labels)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(DATA_DIR, "szegedner"),
        help="Path of the data directory.",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Download dataset even if it exists.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Value of the seed for reproducibility.",
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
        "--valid_idx",
        type=int,
        default=1,
        help="Index of the validation split in from the folds.",
    )

    args = parser.parse_args()

    assert args.valid_idx < args.num_folds

    os.makedirs(args.data_dir, exist_ok=True)

    extract_path = os.path.join(args.data_dir, FILE_NAME)
    download_path = extract_path + ".zip"

    if not os.path.exists(extract_path) or args.force_download:
        if os.path.exists(extract_path):
            shutil.rmtree(extract_path)

        if not os.path.exists(download_path) or args.force_download:
            download(download_path, URL + FILE_NAME + ".zip")

        with zipfile.ZipFile(download_path) as zf:
            zf.extractall(path=extract_path)

    raw_data_path = os.path.join(args.data_dir, FILE_NAME, "hun_ner_corpus.txt")

    train_path = os.path.join(args.data_dir, "train.jsonl")
    valid_path = os.path.join(args.data_dir, "valid.jsonl")

    def write_fold(fold, file_handle, labels):
        for example in fold:
            if example is not None and example.strip() != "":
                labels.update(json.loads(example)["labels"])
                file_handle.write(example + "\n")

    train_labels = collections.Counter()
    valid_labels = collections.Counter()

    with tempfile.TemporaryFile("w+", dir=args.data_dir) as fh:
        for json_example in generate_jsonl(raw_data_path):
            fh.write(json.dumps(json_example) + "\n")

        fh.flush()
        fh.seek(0)

        lines = fh.read().split("\n")
        random.shuffle(lines)

        # computing the fold size note that the last
        # fold might be slightly smaller
        fold_size = math.ceil(len(lines) / args.num_folds)
        folds = group_elements(lines, fold_size)

        with open(train_path, "w") as train_file:
            for idx, fold in enumerate(folds):
                if idx == args.valid_idx:
                    with open(valid_path, "w") as valid_file:
                        write_fold(fold, valid_file, valid_labels)

                else:
                    write_fold(fold, train_file, train_labels)

    labels = {*train_labels, *valid_labels}

    assert min(train_labels.get(l, 0) for l in labels) > args.min_class_example
    assert min(valid_labels.get(l, 0) for l in labels) > args.min_class_example


if __name__ == "__main__":
    main()

