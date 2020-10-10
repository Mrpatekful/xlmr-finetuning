"""
@author:    Patrik Purgai
@copyright: Copyright 2019, xlmr-hungarian
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.04.04.
"""

import os
import argparse
import json
import itertools

from os.path import join, dirname, abspath, exists, basename


def read_file(file_name):
    """
    Reads the lines of the provided file.
    """
    with open(file_name, "r") as fh:
        for line in fh:
            yield line.strip()


def generate_sequences(file_name, delimiter):
    """
    Generates sentences with tags from a file.
    """
    sequence = []

    for line in read_file(file_name):
        split = line.split(delimiter)
        if len(split) > 1:
            # using only the token and ner tag
            # which is the final element in the row
            sequence.append((split[0], split[-1]))
        elif len(sequence) > 0:
            yield list(zip(*sequence))
            sequence = []


def generate_jsonl(file_name, delimiter):
    """
    Converts conll formated file to jsonl.
    """
    for tokens, labels in generate_sequences(file_name, delimiter):
        yield json.dumps({"tokens": tokens, "labels": labels})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path of the data directory."
    )
    parser.add_argument(
        "--delimiter", type=str, default=" ", help="Ner token delimiter."
    )

    args = parser.parse_args()

    splits = [
        f
        for f in os.listdir(args.data_dir)
        if basename(f)[:-4] in ["train", "dev", "test"]
    ]

    for file_name in splits:
        source_path = join(args.data_dir, file_name)

        split_name = basename(file_name)[:-4]

        split_path = join(
            args.data_dir, ("valid" if split_name == "dev" else split_name) + ".jsonl"
        )

        lines = generate_jsonl(source_path, args.delimiter)

        with open(split_path, "w") as fh:
            for line in lines:
                fh.write(line + "\n")


if __name__ == "__main__":
    main()
