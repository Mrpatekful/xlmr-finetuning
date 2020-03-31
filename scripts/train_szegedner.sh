#!/bin/bash

# @author:    Patrik Purgai
# @copyright: Copyright 2019, xlmr-hungarian
# @license:   MIT
# @email:     purgai.patrik@gmail.com
# @date:      2019.07.12.


DATA_DIR="../../../data/szegedner"

if [ ! -d "$DATA_DIR" ]; then
    python $(dirname "$0")/../data/download_szegedner.py
fi

python $(dirname "$0")/train.py data_dir=$DATA_DIR

