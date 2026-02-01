from typing import Callable, Tuple, Optional, Dict

import jax
import jax.numpy as jnp
from flax import linen as nn

class Periodic_Embedding(nn.Module):
    period: float = jnp.pi
    axis: Tuple[int] = (1,)

    @nn.compact
    def __call__(self, x):
        y = []
        for i, xi in enumerate(x):
            if i in self.axis:
                y.extend([jnp.cos(self.period * xi), jnp.sin(self.period * xi)])
            else:
                y.append(xi)

        return jnp.hstack(y)

class Fourier_Embedding(nn.Module):
    scale: float
    dim: int

    @nn.compact
    def __call__(self, x):
        kernel = self.param(
            "kernel", nn.initializers.normal(self.scale), (x.shape[-1], self.dim // 2)
        )
        y = jnp.concatenate(
            [jnp.cos(jnp.dot(x, kernel)), jnp.sin(jnp.dot(x, kernel))], axis=-1
        )
        return y

def weight_fact_init(init_fn, mean, stddev):
    def init(key, shape, dtype=jnp.float32):
        key1, key2 = jax.random.split(key)
        w = init_fn(key1, shape, dtype)
        g = jnp.exp(jax.random.normal(key2, (shape[-1],)) * stddev + mean) # Magnitude of Weights
        v = w / g # Direction of Weights
        return v, g
    return init

class Dense(nn.Module):
    features: int
    kernel_init: callable = nn.initializers.kaiming_normal()
    bias_init: callable = nn.initializers.zeros
    weight_fact: Optional[Dict] = None

    @nn.compact
    def __call__(self, inputs):
        if self.weight_fact is not None:
            mean = self.weight_fact.get('mean', 0.0)
            stddev = self.weight_fact.get('stddev', 0.1)
            v, g = self.param('kernel', weight_fact_init(self.kernel_init, mean, stddev), (inputs.shape[-1], self.features))
            w = v * g
            y = jnp.dot(inputs, w)
        else:
            kernel = self.param('kernel', self.kernel_init, (inputs.shape[-1], self.features))
            y = jnp.dot(inputs, kernel)
        bias = self.param('bias', self.bias_init, (self.features,))
        y = y + bias
        return y
    
class MLP(nn.Module):
    hidden_layers: int = 4
    hidden_size: int = 256
    output_size: int = 1
    activation: Callable = nn.relu
    use_bias: bool = True
    weight_fact: Optional[Dict] = None
    periodic_embed: Optional[Dict] = None
    fourier_embed: Optional[Dict] = None

    @nn.compact
    def __call__(self, x):
        if self.periodic_embed is not None:
            x = Periodic_Embedding(
                period=self.periodic_embed.get('period', jnp.pi),
                axis=self.periodic_embed.get('axis', (1,))
            )(x)

        if self.fourier_embed is not None:
            x = Fourier_Embedding(
                scale=self.fourier_embed.get('scale', 1.0),
                dim=self.fourier_embed.get('dim', 256)
            )(x)

        for _ in range(self.hidden_layers):
            x = Dense(features=self.hidden_size, weight_fact=self.weight_fact)(x)
            x = self.activation(x)
        x = Dense(features=self.output_size, weight_fact=self.weight_fact)(x)
        return x