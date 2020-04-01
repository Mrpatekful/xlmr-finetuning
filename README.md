# Named entity recognition

Named entity recognition with XLM-RoBERTa *[Conneau et al. (2019)](https://arxiv.org/pdf/1911.02116.pdf)* on Hungarian and English datasets. After training the model can be exported to ONNX format for serving with *[ONNX Runtime Server](https://github.com/microsoft/onnxruntime/blob/master/docs/ONNX_Runtime_Server_Usage.md)*.

> NOTE: The model can be trained on any kind of tagging task in any language, but preprocessing scripts are provided for Hungarian named entity recognition datasets like Szeged NER and English CoNNL 2003

## Training

It is convenient to use *[Google Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb)* or *[Kaggle kernel](https://www.kaggle.com/kernels)* as these platforms provide strong GPU-s with half-precision training support. Training XLM-RoBERTa with base configuration on Hungarian Newswire Texts can be performed by the following commands ( assuming the current directory is `xlmr-finetuning/` ). Checkout the below colab link for easily training and evaluating the model.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/Mrpatekful/64f61f1237cb9866d4a6e85e9ea726af/named_entity_recognition.ipynb)
```bash
# downloading the .ner formated corpora at transforming
# it to .jsonl format into data/business folder and extracting
# the labels into labels.json

# python data/download_conll2003.py
python data/download_szegedner.py

# or use your own dataset see the data/ner2jsonl.py script

# downloading the xlmr model from fairseq and serilaizing
# the .jsonl files to .tfrecords
python scripts/train.py data_dir=../../../szegedner
# results are saved into the current working dir under the
# outputs folder
```

To use custom training data such as conll 2003 English ner corpus, place the training, dev and test files ( train.txt, dev.txt and valid.txt with exactly these names ) to a directory in data/ ( `conll2003` for example ) and call `python data/ner2jsonl.py --data_dir data/conll2003`. By running training with the default parameters on the conll2003 files, the model reaches the following performance on the test set. Also with default parameters the model reaces 96.29 F1 score on Szeged NER.

```bash
processed 46435 tokens with 5648 phrases; found: 5757 phrases; correct: 5157.
accuracy:  98.03%; precision:  89.58%; recall:  91.31%; FB1:  90.43
              LOC: precision:  91.86%; recall:  92.03%; FB1:  91.94  1671
             MISC: precision:  78.40%; recall:  83.76%; FB1:  80.99  750
              ORG: precision:  86.67%; recall:  89.28%; FB1:  87.96  1711
              PER: precision:  95.45%; recall:  95.92%; FB1:  95.68  1625
```

## Inference

Currently only ONNX Runtime Server is tested for serving the trained models. To export a checkpointed model, use the `export.py` script.

```bash
# export the model to ONNX format note that onnx==1.6 might
# raise segmentation fault, but that can be ignored
python scripts/export.py ckpt_path=/absolute/path/to/model.pt
```

The inference is actualy performed by two servers, a FastAPI-based tokenizer service, that receives the text from the user, and the ONNX Runtime Server, which receives the tokenized protocol messages from FastAPI server and computes the logits. This is then processed by the FastAPI service and returned to the user.

```bash
# labels path must be available as an env variable
export LABELS_PATH=your/path/to/labels.json

sudo docker run -d -v /your/path/to/model:/your/path/to/model -p 9001:8001 mcr.microsoft.com/onnxruntime/server --model_path /your/path/to/model/xlmr.base.onnx 
```

Or simply use the convenience script.

```bash
./scripts/serve_onnx_runtime.sh /your/abs/path/to/model/xlmr.base.onnx
```
