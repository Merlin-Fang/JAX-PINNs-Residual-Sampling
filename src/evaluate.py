import jax
from jax import vmap
import jax.numpy as jnp
import os

import matplotlib.pyplot as plt

from utils import load_dataset, load_checkpoint

def eval(config, ckptdir, model, step=None):
    u_ref, t, x = load_dataset()

    model.state = load_checkpoint(model.state, ckptdir, step=step)
    params = model.state.params

    u_pred = vmap(vmap(model.get_solution, in_axes=(None, 0, None)), in_axes=(None, None, 0))(params, t, x)
    l2_error = jax.numpy.linalg.norm(u_pred - u_ref) / jax.numpy.linalg.norm(u_ref)
    print("L2 error: {:.3e}".format(l2_error))

    t_grid, x_grid = jax.numpy.meshgrid(t, x, indexing='ij')

    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(t_grid, x_grid, u_ref, cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Reference")
    plt.tight_layout()

    plt.subplot(1, 3, 2)
    plt.pcolor(t_grid, x_grid, u_pred, cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Predicted")
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.pcolor(t_grid, x_grid, jnp.abs(u_ref - u_pred), cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Absolute error")
    plt.tight_layout()

    workdir = '/scratch/merlinf/repos/PINNs-Training-Dynamics'
    save_dir = os.path.join(workdir, "figures", config.experiment)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    fig_path = os.path.join(save_dir, f"{model.state.step}.png")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)