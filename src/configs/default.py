from ml_collections import ConfigDict

def get_config():
    config = ConfigDict()

    config.pde = ConfigDict()
    config.pde.name = 'burgers'
    config.pde.run = '1'
    config.pde.experiment = config.pde.name + '_experiment_' + config.pde.run

    config.model = ConfigDict()
    config.model.hidden_layers = 4
    config.model.hidden_size = 256
    config.model.output_size = 1
    config.model.activation = 'relu'  # Options could be 'relu', 'tanh', etc.
    config.model.weight_fact = None  # Example: {'mean': 0.0, 'stddev': 0.1} or None

    config.training = ConfigDict()
    config.training.seed = 42
    config.training.learning_rate = 1e-3
    config.training.batch_size = 64
    config.training.num_steps = 200000
    config.training.momentum = 0.9
    config.training.loss_weights = {"ic": 1.0, "res": 1.0}  # Initial loss weights can be set here
    config.training.save_freq = None 

    config.weighing = ConfigDict()
    config.weighing.scheme = 'ntk'  # Options could be 'ntk', 'grad_norm', etc.
    config.weighing.momentum = 0.9  # Momentum for updating loss weights
    config.weighing.update_freq = 1000  # Update loss weights every n steps

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