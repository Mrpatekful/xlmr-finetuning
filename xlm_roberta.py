"""
@author:    Patrik Purgai
@copyright: Copyright 2020, xlmr-finetuning
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2020.05.30.
"""

import random
import os
import omegaconf
import torch
import glob
import transformers
import itertools
import datasets
import functools
import re

import pytorch_lightning as pl
import numpy as np

PROJECT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "..")

SPLITS = TRAIN, VALID = datasets.Split.TRAIN, datasets.Split.VALIDATION

BILUO = BEGIN, IN, LAST, UNIT, OUT = "B", "I", "L", "U", "O"


# custom modelcheckpoint is required to overwrite the formatting in the file name
# as default "=" symbol conflicts with hydra's argument parsing
class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def format_checkpoint_name(self, *args, **kwargs):
        return super().format_checkpoint_name(*args, **kwargs).replace("=", ":")


class XlMRobertaModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.model = transformers.GPT2LMHeadModel.from_pretrained(
            self.hparams.pretrained_name
        )

        self.model.resize_token_embeddings(self.hparams.vocab_size)

    def forward(self, batch):
        output = self.model(
            input_ids=batch[INPUT_IDS],
            attention_mask=batch[ATTENTION_MASK],
            return_dict=True,
        )

        logits = output["logits"].view(-1, output["logits"].size(-1))
        label_ids = batch[LABEL_IDS].view(-1)

        loss = torch.nn.functional.cross_entropy(
            logits,
            label_ids,
            ignore_index=self.hparams.pad_id,
        )

        attention_mask = batch[ATTENTION_MASK].view(-1)
        accuracy = (
            (label_ids[attention_mask] == logits[attention_mask].argmax(-1))
            .float()
            .mean()
        )

        ppl = torch.exp(loss)

        return {"loss": loss, "accuracy": accuracy, "ppl": ppl}

    def training_step(self, batch, batch_idx):
        output = self(batch)

        result = pl.TrainResult(output["loss"])
        result.log("loss", output.pop("loss"))

        for name, value in output.items():
            result.log(name, value, prog_bar=True)

        return result

    def validation_step(self, batch, batch_idx):
        output = self(batch)

        result = pl.EvalResult(
            checkpoint_on=output["loss"], early_stop_on=output["loss"]
        )
        result.log("loss", output.pop("loss"))

        for name, value in output.items():
            result.log(name, value, prog_bar=True)

        return result

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

        return optimizer


class XLMRobertaDataModule(pl.LightningDataModule):
    def __init__(self, config, tokenizer):
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer
        self.dataset = None

    def prepare_data(self):
        def run_script(name):
            scripts_dir = os.path.join(PROJECT_DIR, "scripts")
            script = os.path.join(scripts_dir, f"download_{self.config.name}.py")
            params = [
                f"--output_dir {self.config.build_dir}",
                f"--text_field {self.config.text_field}",
                f"--label_field {self.config.label_field}",
            ]

            if self.config.rebuild:
                params.append("--force")

            os.system(f"python {script} {' '.join(params)}")

        if self.config.name is not None:
            run_script(self.config.name)

        # instantiating dataset for building the cache file on a single worker
        build_dataset(self.tokenizer, self.config)

    def setup(self, stage=None):
        self.dataset = build_dataset(self.tokenizer, self.config)

    def train_dataloader(self):
        return build_dataloader(self.dataset[TRAIN], self.config, self.specials)

    def val_dataloader(self):
        return build_dataloader(self.dataset[VALID], self.config, self.specials, False)


def build_dataloader(dataset, config, specials, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset,
        pin_memory=True,
        num_workers=config.num_workers,
        collate_fn=functools.partial(collate, specials=specials),
        batch_sampler=bucket_sampler,
    )


def build_split(
    examples, tokenizer, cache_dir, split_name, tokens_field, labels_field, labels
):
    class_label = datasets.ClassLabel(names=labels)

    examples = examples.map(
        lambda example: build_example(example, tokenizer, tokens_field, labels_field),
        cache_file_name=os.path.join(cache_dir, f"{split_name}.examples.cache"),
        features=datasets.Features({
            labels_field: datasets.Sequence(class_label),
            tokens_field: datasets.Sequence(datasets.Value("int64")),
        }),
    )

    examples.set_format(type="np", columns=examples.columns)

    return examples


def build_dataset(tokenizer, config):
        glob.glob(config.get(str(split)).data_pattern)

    dataset = datasets.load_dataset("json", data_files=data_files)
    labels = build_labels(dataset, config)

    splits = {
        split: build_split(
            examples=examples,
            tokenizer=tokenizer,
            cache_dir=config.cache_dir,
            split_name=split,
            tokens_field=config.tokens_field,
            labels_field=config.labels_field,
            labels=labels,
        )
        for split, examples in dataset.items()
    }

    return splits


def build_labels(dataset, config):
    if os.path.exists(config.labels_file):
        with open(config.labels_file) as fh:
            return {label: idx for idx, label in enumerate(fh.read().split())}
    
    tags = set()

    dataset.map(
        lambda example: tags.update(
            [get_tag(label) for label in example[config.labels_field]]
        )
    )

    tags.remove(OUT)

    labels = [f"{prefix}-{tag}" for prefix, tag in itertools.product(BILUO[:-1], tags)]
    labels.append(OUT)

    with open(config.labels_file, "w") as fh:
        fh.write("\n".join(labels))

    return labels
    

def build_example(example, tokenizer, tokens_field, labels_field, max_tokens):
    sub_token_ids, num_sub_tokens = tokenize(
        example[tokens_field], tokenizer, max_tokens
    )

    extended_labels = extend_labels(example[labels_field], num_sub_tokens)

    return {tokens_field: sub_token_ids, labels_field: extended_labels}


def tokenize(tokens, tokenizer, max_tokens):
    sub_token_ids, num_sub_tokens = [], []

    for token in tokens:
        token_ids = tokenizer(token, max_tokens=max_tokens)["input_ids"]
        sub_token_ids.extend(token_ids)
        num_sub_tokens.append(len(token_ids))

    return np.array(sub_token_ids, dtype=np.int64), num_sub_tokens


def extend_labels(labels, num_sub_tokens):
    def generate_labels():
        for label, num in zip(labels, num_sub_tokens):
            if num == 1:
                yield [label]

            elif label == OUT:
                yield [OUT] * num

            else:
                tag = get_tag(label)
                prefix = get_prefix(label)

                if prefix == BEGIN:
                    yield [f"{BEGIN}-{tag}"] + build_inside_tags(tag, num - 1)

                elif prefix == IN:
                    yield build_inside_tags(tag, num)

                elif prefix == UNIT:
                    inside_tags = build_inside_tags(tag, num - 2)
                    yield [f"{BEGIN}-{tag}"] + inside_tags + [f"{LAST}-{tag}"]

                else:
                    yield build_inside_tags(tag, num - 1) + [f"{LAST}-{tag}"]

    extended_labels = itertools.chain(*generate_labels())

    return extended_labels


def get_tag(label):
    return label.split("-")[-1]


def get_prefix(label):
    return label.split("-")[0]


def build_inside_tags(tag, num):
    return ["f{IN}-{tag}" for _ in range(num)]


def collate(batch, specials):
    pad_id, bot_id, usr_id = specials

    batch_size = len(batch)
    max_len = max([e[INPUT_IDS].shape[0] for e in batch]) - 1

    input_ids = np.full((batch_size, max_len), pad_id, dtype=np.int64)
    label_ids = np.copy(input_ids)
    attention_mask = np.zeros_like(input_ids, dtype=np.int8)

    for idx, example in enumerate(batch):
        example_len = example[INPUT_IDS].shape[0] - 1
        input_ids[idx, :example_len] = example[INPUT_IDS][:-1]
        attention_mask[idx, :example_len] = example[ATTENTION_MASK][:-1]

        label_ids[idx, :example_len] = example[INPUT_IDS][1:]

    label_ids = torch.as_tensor(label_ids)
    label_ids[(label_ids == bot_id) | (label_ids == usr_id)] = pad_id

    return {
        INPUT_IDS: torch.as_tensor(input_ids),
        LABEL_IDS: label_ids,
        ATTENTION_MASK: torch.as_tensor(attention_mask, dtype=torch.bool),
    }
