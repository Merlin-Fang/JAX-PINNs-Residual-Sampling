from ml_collections import ConfigDict
from jax import numpy as jnp

def get_config():
    config = ConfigDict()

    config.pde = ConfigDict()
    config.pde.name = 'burgers'
    config.pde.run = 'NTK+ResSampler'
    config.pde.experiment = config.pde.name + '_experiment_' + config.pde.run

    config.model = ConfigDict()
    config.model.hidden_layers = 4
    config.model.hidden_size = 256
    config.model.output_size = 1
    config.model.activation = 'tanh'  # Options could be 'relu', 'tanh', etc.
    config.model.weight_fact = {'mean': 0.5, 'stddev': 0.1}
    config.model.periodic_embed = {'period': jnp.pi, 'axis': (1,)}
    config.model.fourier_embed = {'scale': 1.0, 'dim': 256}

    config.training = ConfigDict()
    config.training.seed = 42
    config.training.batch_size = 4096 * 4 # Total batch size across all devices
    config.training.num_steps = 50000
    config.training.momentum = 0.9
    config.training.loss_weights = {"ic": 1.0, "res": 1.0}  # Initial loss weights can be set here
    config.training.save_freq = None 

    config.optim = ConfigDict()
    config.optim.grad_accum_steps = 0
    config.optim.optimizer = "Adam"
    config.optim.beta1 = 0.9
    config.optim.beta2 = 0.999
    config.optim.eps = 1e-8
    config.optim.learning_rate = 3e-3
    config.optim.decay_rate = 0.9
    config.optim.decay_steps = 2000

    config.weighing = ConfigDict()
    config.weighing.scheme = 'ntk'  # Options could be 'ntk', 'grad_norm', etc.
    config.weighing.momentum = 0.9  # Momentum for updating loss weights
    config.weighing.update_freq = 1000  # Update loss weights every n steps

    config.sampling = ConfigDict()
    config.sampling.method = 'residual'  # Options: 'uniform', 'residual'
    config.sampling.pool_size = 4096 * 5
    config.sampling.temperature = 1.0
    config.sampling.uniform_eps = 0.1
    config.sampling.recalcprob_freq = 100 # Frequency to recalculate pool probabilities
    config.sampling.refpool_freq = 1000 # Frequency to refresh the candidate pool

    config.wandb = ConfigDict()
    config.wandb.use = False
    config.wandb.project = 'pinns-training-dynamics'

    config.logging = ConfigDict()
    config.logging.handler_type = 'file'  # Options: 'stream' or 'file'
    config.logging.log_dir = '/scratch/merlinf/repos/PINNs-Training-Dynamics/pdes/' + config.pde.name + '/logs/'
    config.logging.freq = 100
    config.logging.log_losses = True
    config.logging.log_loss_weights = True
    config.logging.log_ntk = True
    config.logging.log_l2_error = True

    return config