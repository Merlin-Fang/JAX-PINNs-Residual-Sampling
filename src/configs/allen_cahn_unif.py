from ml_collections import ConfigDict
from jax import numpy as jnp

def get_config():
    config = ConfigDict()

    config.pde = ConfigDict()
    config.pde.name = 'allen_cahn'
    config.pde.run = 'uniform_sampling'
    config.pde.experiment = config.pde.name + '_' + config.pde.run

    config.model = ConfigDict()
    config.model.hidden_layers = 4
    config.model.hidden_size = 256
    config.model.output_size = 1
    config.model.activation = 'tanh'
    config.model.weight_fact = {'mean': 0.5, 'stddev': 0.1}
    config.model.periodic_embed = {'period': jnp.pi, 'axis': (1,)}
    config.model.fourier_embed = {'scale': 1.0, 'dim': 256}

    config.training = ConfigDict()
    config.training.seed = 42
    config.training.global_batch_size = 4096
    config.training.batch_size_per_device = 1024
    config.training.num_steps = 200000
    config.training.momentum = 0.9
    config.training.loss_weights = {"ic": 1.0, "res": 1.0}
    config.training.save_freq = 1000 

    config.optim = ConfigDict()
    config.optim.grad_accum_steps = 0
    config.optim.optimizer = "Adam"
    config.optim.beta1 = 0.9
    config.optim.beta2 = 0.999
    config.optim.eps = 1e-8
    config.optim.learning_rate = 1e-3
    config.optim.decay_rate = 0.9
    config.optim.decay_steps = 2000

    config.weighing = ConfigDict()
    config.weighing.scheme = 'ntk'
    config.weighing.momentum = 0.9
    config.weighing.update_freq = 1000

    config.sampling = ConfigDict()
    config.sampling.method = 'uniform'
    config.sampling.hard_bank_mult = 50
    config.sampling.cand_mult = 50
    config.sampling.top_frac = 0.05
    config.sampling.hardness = "abs"
    config.sampling.refresh_freq = 200

    config.wandb = ConfigDict()
    config.wandb.use = False
    config.wandb.project = 'PINNs-Residual-Sampling'

    config.logging = ConfigDict()
    config.logging.handler_type = 'file'
    config.logging.log_dir = '/scratch/merlinf/repos/PINNs-Training-Dynamics/pdes/' + config.pde.name + '/logs/'
    config.logging.freq = 100
    config.logging.log_losses = True
    config.logging.log_loss_weights = True
    config.logging.log_ntk = True
    config.logging.log_l2_error = True

    return config