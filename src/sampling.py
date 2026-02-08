import jax
import jax.numpy as jnp
from jax import random, vmap, pmap, local_device_count
from functools import partial

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
    Hybrid sampler:
      - each batch: (1-p) uniform + p hard (sampled uniformly from a persistent hard_bank)
      - hard_bank updated via refresh(params) called explicitly from train.py

    residual_fn must be scalar-per-point:
        residual_fn(params, t, x) -> scalar
    (Matches your PINNs.get_residual signature.)
    """

    def __init__(
        self,
        dom,
        batch_size,
        residual_fn,
        rng_key: int = 42,
        *,
        p: float = 0.0,            # will be updated by train.py
        hard_bank_mult: int = 50,  # hard bank size H = hard_bank_mult * B_global
        cand_mult: int = 100,      # candidate pool size N = cand_mult * B_global (rounded down to multiple of B_global)
        top_frac: float = 0.10,    # select from top top_frac of candidates
        hardness: str = "abs",     # "abs" or "sq"
    ):
        super().__init__(dom, batch_size, rng_key)

        self.p = float(p)
        self.residual_fn = residual_fn

        self.B_global = self.batch_size * self.num_devices
        self.H = int(hard_bank_mult * self.B_global)

        self.cand_mult = int(cand_mult)
        self.top_frac = float(top_frac)

        if hardness not in ("abs", "sq"):
            raise ValueError("hardness must be 'abs' or 'sq'")
        self.hardness = hardness

        self.key, k0 = random.split(self.key)
        self.hard_bank = random.uniform(
            k0,
            shape=(self.H, self.dim),
            minval=self.dom[:, 0],
            maxval=self.dom[:, 1],
        )

        @partial(jax.pmap, axis_name="devices")
        def _hardness_pmap(params, batch):
            t = batch[:, 0]
            x = batch[:, 1]
            r = vmap(self.residual_fn, in_axes=(None, 0, 0))(params, t, x)  # (Bd,)
            if self.hardness == "sq":
                return r * r
            return jnp.abs(r)

        self._hardness_pmap = _hardness_pmap

    def data_generation(self, keys: jnp.ndarray) -> jnp.ndarray:
        """
        keys: (num_devices,) PRNG keys
        returns: (num_devices, batch_size, dim)
        """
        n_hard = int(round(self.p * self.batch_size))
        n_hard = max(0, min(self.batch_size, n_hard))
        n_uni = self.batch_size - n_hard

        def sample_one(k):
            k1, k2, k3 = random.split(k, 3)

            uni = random.uniform(
                k1,
                shape=(n_uni, self.dim),
                minval=self.dom[:, 0],
                maxval=self.dom[:, 1],
            )

            if n_hard > 0:
                idx = random.randint(k2, shape=(n_hard,), minval=0, maxval=self.H)
                hard = self.hard_bank[idx]  # (n_hard, dim)
                batch = jnp.concatenate([uni, hard], axis=0)
            else:
                batch = uni

            perm = random.permutation(k3, self.batch_size)
            return batch[perm]

        return vmap(sample_one)(keys)
    
    def refresh(self, params):
        """
        params: replicated params pytree (e.g., model.state.params)

        Steps:
          1) sample N candidates uniformly
          2) compute hardness for all candidates
          3) take top-k and insert into hard_bank (replace random entries)
        """
        if self.p <= 0.0:
            return
        
        N = int(self.cand_mult * self.B_global)

        self.key, kc = random.split(self.key)
        cand = random.uniform(
            kc,
            shape=(N, self.dim),
            minval=self.dom[:, 0],
            maxval=self.dom[:, 1],
        )

        hardness = self._eval_hardness_in_chunks(params, cand)  # (N,)

        k = max(1, int(round(self.top_frac * N)))

        top_idx = jnp.argpartition(hardness, -k)[-k:]
        new_hard = cand[top_idx]  # (k, dim)

        self.key, kr = random.split(self.key)
        repl_idx = random.randint(kr, shape=(k,), minval=0, maxval=self.H)
        self.hard_bank = self.hard_bank.at[repl_idx].set(new_hard)

    def _eval_hardness_in_chunks(self, params, cand: jnp.ndarray) -> jnp.ndarray:
        """
        cand: (N, dim) with N divisible by B_global
        returns: (N,) hardness
        """
        N = int(cand.shape[0])
        chunk = self.B_global
        D = self.num_devices
        Bd = self.batch_size

        out = []
        for start in range(0, N, chunk):
            block = cand[start:start + chunk]
            block = block.reshape(D, Bd, self.dim)
            h = self._hardness_pmap(params, block)
            out.append(h.reshape(-1))

        return jnp.concatenate(out, axis=0)
