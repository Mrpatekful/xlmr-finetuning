"""
@author:    Patrik Purgai
@copyright: Copyright 2019, xlmr-finetuning
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.09.30.
"""

import os
import sys
import json
import httpx

import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel
from nltk.tokenize import RegexpTokenizer

from os.path import (
    join, dirname,
    abspath, exists)

PROJECT_DIR = join(abspath(dirname(__file__)), '..')
SERVING_DIR = join(PROJECT_DIR, 'src', 'serving')
# local path must be in sys so protobuf imports
# can read local directory

if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

if SERVING_DIR not in sys.path:
    sys.path.append(SERVING_DIR)

from src.data import (
    encode_example, decode_example)

from src.model import create_pretrained

from src.serving.onnx_ml_pb2 import TensorProto
from src.serving.predict_pb2 import (
    PredictResponse, PredictRequest)


class Query(BaseModel):
    text: str
    

app = FastAPI()

URL = os.environ.get('URL', 'http://127.0.0.1')

# change appropriately if needed based on any changes
# invoking the server
PORT = os.environ.get('PORT', '9001')

MODEL_TYPE = os.environ.get('MODEL_TYPE', 'xlmr.base')

LABELS_PATH = os.environ.get('LABELS_PATH', 'labels.json')

assert exists(LABELS_PATH)

with open(LABELS_PATH, 'r') as fh:
    label2id = json.load(fh)

# creating inverse lookup for decoding
id2label = {v: k for k, v in label2id.items()}

xlmr = create_pretrained(MODEL_TYPE)

# the model can be deleted from the memory as
# only the task dictionary is required
del xlmr.model


tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')


@app.post('/')
async def predict(query: Query):
    """
    Calls the ONNX Runtime Server with the tokenized
    query string and returns the predictions.
    """
    tokens = tokenizer.tokenize(query.text)

    # this is only required for the tokenizer function
    # so the predictions can be restored from logits
    labels = ['O'] * len(tokens)

    input_ids, label_ids = encode_example(
        tokens=tokens, labels=labels,
        xlmr=xlmr, label2id=label2id)

    input_array = np.array(input_ids, dtype=np.int64)
    input_array = np.expand_dims(input_array, 0)

    input_tensor = TensorProto()
    input_tensor.dims.extend(input_array.shape)
    # 7 is INTS from onnx ml proto attributes
    input_tensor.data_type = 7
    input_tensor.raw_data = input_array.tobytes()

    request_message = PredictRequest()

    request_message.inputs['input'].data_type = \
        input_tensor.data_type

    request_message.inputs['input'].dims.extend(
        input_tensor.dims)

    request_message.inputs['input'].raw_data = \
        input_tensor.raw_data

    content_type_headers = [
        'application/x-protobuf',
        'application/octet-stream',
        'application/vnd.google.protobuf'
    ]

    for header in content_type_headers:
        request_headers = {
            'Content-Type': header,
            'Accept': 'application/x-protobuf'
        }

    url = '{}:{}/v1/models/default/versions/1:predict'

    # fetching the logits from the model
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url.format(URL, PORT),
            headers=request_headers,
            data=request_message.SerializeToString())

    response_message = PredictResponse()
    response_message.ParseFromString(response.content)

    output = response_message.outputs['output'].raw_data

    logits = np.frombuffer(output, dtype=np.float32)
    logits = np.reshape(logits, (-1, len(label2id)))

    pred_ids = np.argmax(logits, axis=-1).tolist()

    pred_ids = [
        (pred if label != -1 else -1)
        for pred, label in
        zip(pred_ids, label_ids)
    ]

    _, preds = decode_example(
        input_ids, pred_ids,
        xlmr=xlmr, id2label=id2label)

    return {'preds': list(zip(tokens, preds))}

