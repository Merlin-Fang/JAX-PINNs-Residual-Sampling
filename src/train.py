import os
import time

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

import ml_collections
import wandb

from sampling import Sampler
from logging import Logger
from pdes.burgers.model import Burgers

from utils import load_dataset, save_checkpoint

def train(config: ml_collections.ConfigDict):
    if config.wandb.use:
        wandb.init(
            project=config.wandb.project,
            name=config.experiment,
        )
    
    logger = Logger(
        name=config.experiment,
        handler_type=config.logging.handler_type,
        log_info={
            'log_dir': config.logging.log_dir,
            'file_name': config.experiment,
        } if config.logging.handler_type == 'file' else None,
    )

    u_ref, t, x = load_dataset()

    model = Burgers(config, IC=(u_ref[0, :], jnp.full_like(x, t[0]), x))
    sampler = Sampler(jnp.array([[t[0], t[-1]], [x[0], x[-1]]]), config.training.batch_size, config.training.seed)

