"""
@author:    Patrik Purgai
@copyright: Copyright 2020, hugpt
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2020.05.30.
"""

import os
import pprint
import torch
import hydra
import logging
import warnings
import omegaconf
import transformers
import gpt2

import pytorch_lightning as pl

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity(transformers.logging.WARNING)
pl._logger.handlers = []


@hydra.main(config_name="config.yaml")
def main(config: omegaconf.OmegaConf):
    assert config.checkpoint_file is not None

    logging.info("\n" + config.pretty())

    config.output_dir = os.path.expanduser(config.output_dir)
    pl.trainer.seed_everything(config.seed)

    model = gpt2.GPT2Module.load_from_checkpoint(
        config.checkpoint_file, history_size=config.history_size
    )

    with torch.no_grad():
        pprint.pprint(model.decode([["asd", "asd asdas"]]))


if __name__ == "__main__":
    main()
