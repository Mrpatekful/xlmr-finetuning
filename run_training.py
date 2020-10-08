"""
@author:    Patrik Purgai
@copyright: Copyright 2020, hugpt
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2020.05.30.
"""

import os
import torch
import hydra
import logging
import warnings
import typing
import omegaconf
import transformers
import gpt2

import pytorch_lightning as pl

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity(transformers.logging.WARNING)
pl._logger.handlers = []


@hydra.main(config_name="config.yaml")
def main(config: omegaconf.OmegaConf):
    config.output_dir = os.path.expanduser(config.output_dir)
    pl.trainer.seed_everything(config.seed)

    logging.info("\n" + config.pretty())

    data_module = gpt2.GPT2DataModule(config)
    data_module.prepare_data()

    if config.checkpoint_file is not None:
        trainer = pl.Trainer(resume_from_checkpoint=config.checkpoint_file)
        model = gpt2.GPT2Module.load_from_checkpoint(config.checkpoint_file)

    else:
        hparams = {
            "tokenizer_dir": data_module.output_dir,
            "pretrained_name": config.pretrained_name,
            "history_size": config.history_size,
            **config.hparams,
        }

        model = gpt2.GPT2Module(**hparams)

        model_checkpoint = gpt2.ModelCheckpoint("{epoch}-{loss:.2f}", save_top_k=1)
        early_stopping = pl.callbacks.EarlyStopping()
        gpu_stats = pl.callbacks.GPUStatsMonitor()

        trainer = pl.Trainer(
            gpus=config.gpus,
            num_nodes=config.num_nodes,
            checkpoint_callback=model_checkpoint,
            early_stop_callback=early_stopping,
            callbacks=[gpu_stats],
        )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
