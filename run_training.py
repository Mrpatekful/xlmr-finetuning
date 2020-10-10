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
import xlm_roberta

import pytorch_lightning as pl

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity(transformers.logging.WARNING)
pl._logger.handlers = []


@hydra.main(config_path="config", config_name="training")
def main(config: omegaconf.OmegaConf):
    pl.trainer.seed_everything(config.seed)

    logging.info("\n" + config.pretty())

    tokenizer = transformers.XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

    data_module = xlm_roberta.XLMRobertaDataModule(config.data, tokenizer)
    data_module.prepare_data()

    if config.checkpoint_file is not None:
        trainer = pl.Trainer(resume_from_checkpoint=config.checkpoint_file)
        model = xlm_roberta.XLMRobertaModule.load_from_checkpoint(config.checkpoint_file)

    else:
        model = xlm_roberta.XLMRobertaModule(**hparams)

        model_checkpoint = xlm_roberta.ModelCheckpoint("{epoch}-{loss:.2f}", save_top_k=1)
        early_stopping = pl.callbacks.EarlyStopping()

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
