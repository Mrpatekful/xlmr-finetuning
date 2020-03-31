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
import requests
import itertools

import ner2jsonl

from os.path import (
    join, dirname,
    abspath, exists,
    basename)


FILES = [
    ('1fu9ZiFnn2U5c1t0rGoT2EMJ6OfuEUfse', 'train.txt'),
    ('1D6S3yrVKFcgcs3P4zXN935PPjd_UdjxN', 'test.txt'),
    ('1dqa-aBgqZn6r0itafAGwkANVOlvk825a', 'dev.txt')
]


def download_file(file_id, dest):
    """
    Downloads a file from google drive.
    """
    URL = 'https://docs.google.com/uc?export=download'

    with requests.Session() as session:

        response = session.get(
            URL, params={'id': file_id}, stream=True)

        token = get_confirm_token(response)

        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(
                URL, params=params, stream=True)

    write_file(response, dest)


def get_confirm_token(response):
    """
    Gets the confirm token from the response.
    """
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def write_file(response, dest):
    """
    Writes the content of the response to file.
    """
    CHUNK_SIZE = 32768

    with open(dest, 'wb') as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default=join(abspath(dirname(__file__)), 'conll2003'),
        help='Path of the data directory.')
    
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    for file_id, split_name in FILES:
        download_file(
            file_id, join(args.data_dir, split_name))

    splits = [
        f for f in os.listdir(args.data_dir)
        if basename(f)[:-4] in ['train', 'dev', 'test']
    ]

    for file_name in splits:
        source_path = join(args.data_dir, file_name)
        split_name = basename(file_name)[:-4]

        split_path = join(
            args.data_dir,
            ('valid' if split_name == 'dev' else split_name) \
            + '.jsonl')

        lines = ner2jsonl.generate_jsonl(
            source_path, delimiter=' ')

        with open(split_path, 'w') as fh:
            for line in lines:
                fh.write(line + '\n')


if __name__ == '__main__':
    main()

