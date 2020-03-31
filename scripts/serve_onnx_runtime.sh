#!/bin/bash

# author:    Patrik Purgai
# copyright: Copyright 2019, xlmr-hungarian
# license:   MIT
# email:     purgai.patrik@gmail.com
# date:      2019.04.04.

MODEL_PATH=${1:-"xlmr.base.onnx"}
MODEL_DIR=$(dirname "$MODEL_PATH")
LABELS_PATH=${2:-$MODEL_DIR/labels.json}

# MODEL_PATH must be an absolute path
sudo docker run -d -v $MODEL_DIR:$MODEL_DIR -p 9001:8001 mcr.microsoft.com/onnxruntime/server --model_path $MODEL_PATH

export LABELS_PATH=$LABELS_PATH

uvicorn scripts.serve:app
