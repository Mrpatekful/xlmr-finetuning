"""
@author:    Patrik Purgai
@copyright: Copyright 2020, dialoue-generation
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2020.05.30.
"""

import typing
import os
import omegaconf
import torch
import glob
import transformers

import pytorch_lightning as pl
import tensorflow as tf

PROJECT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..")

NO_LABEL = -1


class Batch(typing.NamedTuple):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def format_checkpoint_name(
        self, *args: typing.Any, **kwargs: typing.Any
    ) -> typing.Text:
        return super().format_checkpoint_name(*args, **kwargs).replace("=", ":")


class XLMRobertaModule(pl.LightningModule):
    def __init__(
        self,
        tokenizer_dir: typing.Text,
        pretrained_name: typing.Text = "xlm-roberta-base",
        lr: float = 1e-3,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = transformers.XLMRobertaForTokenClassification.from_pretrained(
            self.hparams.pretrained_name
        )
        self.tokenizer = transformers.XLMRobertaTokenizer.from_pretrained(
            self.hparams.tokenizer_dir
        )

        self.pad_id = self.tokenizer.pad_id

    def forward(self, batch: Batch) -> torch.Tensor:
        out = self.model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            return_dict=True,
        )

        loss = torch.nn.functional.cross_entropy(
            out["logits"].view(-1, out["logits"].size(-1)),
            batch.labels.view(-1),
            ignore_index=NO_LABEL,
        )

        return loss

    def training_step(self, batch: Batch, batch_idx: int) -> pl.TrainResult:
        loss = self(batch)

        result = pl.TrainResult(loss)

        return result

    def validation_step(self, batch: Batch, batch_idx: int) -> pl.EvalResult:
        loss = self(batch)

        result = pl.EvalResult(checkpoint_on=loss, early_stop_on=loss)

        return result

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(self.parameters())

        return optimizer


class CONLLDataModule(pl.LightningDataModule):
    def __init__(self, config: omegaconf.DictConfig):
        super().__init__()

        self.config = config
        self.train = None
        self.valid = None
        self.test = None

    def prepare_data(self) -> typing.NoReturn:
        output_dir = self.output_dir

        try:
            transformers.XLMRobertaTokenizer.from_pretrained(output_dir)

        except OSError:
            tokenizer = transformers.XLMRobertaTokenizer.from_pretrained(
                self.config.pretrained_name
            )

            tokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)
            tokenizer.save_pretrained(output_dir)

        scripts_dir = os.path.join(PROJECT_DIR, "scripts")

        os.system(
            f"python {os.path.join(scripts_dir, 'download_')}"
            f"{self.config.data}.py --output_dir {output_dir}"
        )

        for split in ["train", "valid"]:
            os.system(
                f"python {os.path.join(scripts_dir, 'build_tfrecords_from_text.py')} "
                f"--tokenizer {output_dir} "
                f"--pattern {os.path.join(output_dir, split)}.txt "
                f"--output_dir {os.path.join(output_dir, split)} "
                f"--pretrained {self.config.pretrained_name}"
            )

    def setup(self, stage: typing.Optional[typing.Text] = None) -> typing.NoReturn:
        tokenizer = transformers.XLMRobertaTokenizer.from_pretrained(self.output_dir)

        self.train = TaggingDataset(self.config, tokenizer)
        self.valid = TaggingDataset(self.config, tokenizer)
        self.test = TaggingDataset(self.config, tokenizer)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.train, pin_memory=True)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.valid, pin_memory=True)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.test, pin_memory=True)


class TaggingDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        config: omegaconf.DictConfig,
        pattern: typing.Text,
        tokenizer: transformers.GPT2Tokenizer,
    ):
        self.config = config
        self.pattern = pattern
        self.tokenizer = tokenizer

    def __iter__(self):
        tfrecord_pattern = os.path.join(get_output_dir(self.config), self.pattern)

        dataset = tf.data.TFRecordDataset(glob.glob(tfrecord_pattern))

        def preprocess_example(example):
            sequence_features = {"dialogue": tf.io.VarLenFeature(dtype=tf.int64)}

            _, dialogue = tf.io.parse_single_sequence_example(
                example, sequence_features=sequence_features
            )

            size = tf.shape(dialogue["dialogue"])[0]

            dialogue = tf.RaggedTensor.from_sparse(dialogue["dialogue"])

            indices = tf.map_fn(
                lambda idx: tf.RaggedTensor.from_tensor(
                    tf.expand_dims(
                        tf.range(tf.maximum(0, idx - history_size - 1), idx + 1), 0
                    )
                ),
                tf.range(1, size),
                fn_output_signature=tf.RaggedTensorSpec(
                    shape=[1, None], dtype=tf.int32
                ),
            ).merge_dims(-2, -1)

            role_ids = tf.map_fn(
                lambda idx: tf.RaggedTensor.from_tensor(
                    tf.fill([1, tf.size(dialogue[idx])], idx % 2)
                ),
                tf.range(size),
                fn_output_signature=tf.RaggedTensorSpec(
                    shape=[1, None], dtype=tf.int32
                ),
            ).merge_dims(-2, -1)

            role_ids = tf.gather(role_ids, indices.flat_values)
            role_ids = tf.RaggedTensor.from_row_splits(role_ids, indices.row_splits)
            role_ids = tf.cast(role_ids.merge_dims(-2, -1), tf.bool)

            role_ids = tf.map_fn(
                lambda tensor: tf.RaggedTensor.from_tensor(
                    tf.expand_dims(
                        tf.where(
                            tf.cond(tensor[-1], lambda: ~tensor, lambda: tensor),
                            usr_id,
                            bot_id,
                        ),
                        0,
                    ),
                ),
                role_ids,
                fn_output_signature=tf.RaggedTensorSpec(
                    shape=[1, None], dtype=tf.int64
                ),
            ).merge_dims(-2, -1)

            input_ids = tf.gather(dialogue, indices.flat_values)
            input_ids = tf.RaggedTensor.from_row_splits(input_ids, indices.row_splits)
            input_ids = tf.cast(input_ids.merge_dims(-2, -1), tf.int64)

            return {
                "input_ids": input_ids[:, :-1],
                "role_ids": role_ids[:, :-1],
                "labels": input_ids[:, 1:],
            }

        def postprocess_example(example):
            example = {
                name: tensor.to_tensor(default_value=pad_id)
                for name, tensor in example.items()
            }

            example["attention_mask"] = example["input_ids"] != pad_id

            return example

        dataset = (
            dataset.map(
                preprocess_example,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            .unbatch()
            .shuffle(5000)
            .apply(
                tf.data.experimental.bucket_by_sequence_length(
                    lambda x: tf.size(x["input_ids"]), [50], [2, 2], no_padding=True
                )
            )
            .map(postprocess_example)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        for batch, idx in zip(dataset.as_numpy_iterator(), range(10)):
            yield Batch(**batch)


def get_output_dir(config: omegaconf.DictConfig):
    return os.path.join(config.output_dir, config.data)

