import os
import jax
import scipy.io

from jax.tree_util import tree_map
from flax.training import checkpoints

def load_dataset():
    data = scipy.io.loadmat("/scratch/merlinf/repos/PINNs-Training-Dynamics/pdes/burgers/data/burgers.mat")
    u_ref = data["usol"]
    t = data["t"].flatten()
    x = data["x"].flatten()
    return u_ref, t, x

def save_checkpoint(state, step, ckpt_dir):
    if os.path.exists(ckpt_dir) == False:
        os.makedirs(ckpt_dir) 
    
    state = jax.device_get(tree_map(lambda x: x[0], state))
    step = int(jax.device_get(step))

    checkpoints.save_checkpoint(ckpt_dir, state, step, keep=5)

def load_checkpoint(state, ckpt_dir, step = None):
    return checkpoints.restore_checkpoint(ckpt_dir, state, step=step)