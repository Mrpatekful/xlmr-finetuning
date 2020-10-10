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
        yield {"tokens": tokens, "labels": labels}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(DATA_DIR, "szegedner"),
        help="Path of the data directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download and prepare the dataset even if it exists.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    output_path = os.path.join(args.output_dir, "data.jsonl")

    if not os.path.exists(output_path) or args.force_download:
        with tempfile.TemporaryDirectory(dir=args.output_dir) as td:
            download_path = os.path.join(td, FILE_NAME + ".zip")

            download(download_path, URL + FILE_NAME + ".zip")

            with zipfile.ZipFile(download_path) as zf:
                zf.extractall(path=td)

            raw_data_path = os.path.join(td, "hun_ner_corpus.txt")

            labels = collections.Counter()

            with open(output_path, "w") as fh:
                for json_example in generate_jsonl(raw_data_path):
                    fh.write(json.dumps(json_example) + "\n")


if __name__ == "__main__":
    main()
