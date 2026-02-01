from jax import grad
import jax.numpy as jnp

from src.basemodel import PINNs

class Allen_Cahn(PINNs):
    """
    Allen–Cahn Equation: u_t = a*u + v*u_xx - a*u^3
    a = 5
    v = 0.0001
    """
    def __init__(self, config, IC):
        super().__init__(config, IC)
        self.a = 5.0
        self.v = 1e-4

    def get_residual(self, params, t, x):
        u = self.get_solution(params, t, x)
        u_t = grad(self.get_solution, argnums=1)(params, t, x)
        u_xx = grad(grad(self.get_solution, argnums=2), argnums=2)(params, t, x)

        residual = u_t - self.a * u - self.v * u_xx + self.a * (u ** 3)
        return residual
