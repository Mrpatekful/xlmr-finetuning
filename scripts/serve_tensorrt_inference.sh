#!/bin/bash

docker run --rm --runtime=nvidia --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 -v/home/patrik/Documents/tensorrt-inference-server/docs/examples/model_repository:/models nvcr.io/nvidia/tensorrtserver:19.12-py3 trtserver --model-repository=/models

