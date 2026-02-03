import jax
import jax.numpy as jnp
from jax import random, vmap, local_device_count

from torch.utils.data import Dataset

class BaseSampler(Dataset):
    def __init__(self, dom, batch_size, rng_key=42):
        self.key = random.PRNGKey(rng_key)
        self.batch_size = batch_size
        self.num_devices = local_device_count()
        self.dom = dom
        self.dim = dom.shape[0]

    def data_generation(self, keys):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def __getitem__(self, index):
        self.key, key1 = random.split(self.key)
        keys = random.split(key1, self.num_devices)
        batch = self.data_generation(keys)
        return batch


class UniformSampler(BaseSampler):
    def __init__(self, dom, batch_size, rng_key=42):
        super().__init__(dom, batch_size, rng_key)

    def data_generation(self, keys):
        sample_one = lambda k: random.uniform(
            k,
            shape=(self.batch_size, self.dim),
            minval=self.dom[:, 0],
            maxval=self.dom[:, 1],
        )
        return vmap(sample_one)(keys)
    

class ResSampler(BaseSampler):
    """
    Residual-adaptive sampler.

    Keeps a cached candidate pool (self.pool) and sampling probabilities (self.prob).
    - data_generation(): draws training batches from the cached pool using self.prob
    - update_pool(): refreshes (pool, prob) based on current model residuals

    You must call update_pool(params, ...) periodically from the training loop
    once the model exists.
    """

    def __init__(
        self,
        dom,
        batch_size,
        res_fn,
        rng_key=42,
        pool_size=40960,
        temperature=1.0,
        uniform_eps=0.1
    ):
        super().__init__(dom, batch_size, rng_key)
        self.res_fn = vmap(res_fn, in_axes=(None, 0, 0))
        self.pool_size = int(pool_size)
        self.temperature = float(temperature)
        self.uniform_eps = float(uniform_eps)

        self.pool = None  # (pool_size, dim)
        self.prob = None  # (pool_size,)

        self.key, k = random.split(self.key)
        self._refresh_uniform_pool(k)

    def data_generation(self, keys):
        pool = self.pool
        prob = self.prob

        def sample_one_device(k):
            idx = random.choice(
                k,
                a=self.pool_size,
                shape=(self.batch_size,),
                replace=True,
                p=prob,
            )
            return pool[idx, :]

        return vmap(sample_one_device)(keys)

    def update_pool(self, params, refresh_pool):
        if refresh_pool:
            self.key, key = random.split(self.key)
            self._refresh_uniform_pool(key)

        r = self.res_fn(params, self.pool[:, 0], self.pool[:, 1])
        score = jnp.abs(r)
        score = jnp.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

        # If everything is zero, fall back to uniform
        score_sum = jnp.sum(score)
        def make_uniform():
            return jnp.ones((self.pool_size,), dtype=jnp.float32) / self.pool_size

        def make_soft_prob():
            mean = jnp.mean(score) + 1e-8
            s = score / mean 

            delta = 1e-6
            alpha = 1.0 / max(self.temperature, 1e-6)

            p = jnp.power(s + delta, alpha)

            p = (1.0 - self.uniform_eps) * p + self.uniform_eps * (1.0 / self.pool_size)

            p = p / (jnp.sum(p) + 1e-12)
            return p.astype(jnp.float32)

        self.prob = jax.lax.cond(score_sum <= 1e-12, make_uniform, make_soft_prob)

    def _refresh_uniform_pool(self, key):
        self.pool = random.uniform(
            key,
            shape=(self.pool_size, self.dim),
            minval=self.dom[:, 0],
            maxval=self.dom[:, 1],
        )
        self.prob = jnp.ones((self.pool_size,), dtype=jnp.float32) / self.pool_size
