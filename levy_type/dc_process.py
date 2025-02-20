from dataclasses import dataclass

import numpy as np

from levy_type.base_process import BaseProcess
from levy_type.simulation_config import SimulationConfigDC


@dataclass
class DCProcess(BaseProcess):
    config: SimulationConfigDC

    @staticmethod
    def drift_coefficient(t: float, x: float) -> float:
        return np.sin(x)

    @staticmethod
    def diffusion_coefficient(t: float, x: float) -> float:
        return 0

    @staticmethod
    def jump_coefficient(t: float, x: float, z: float) -> float:
        return np.cos(x) * z

    def tau(self, t: float) -> float:
        alpha = self.config.alpha
        return ((alpha + t) / t) ** (-1 / alpha)

    def inverse_large_jump_cdf(self, u: float, t: float) -> float:
        h, eps = self.config.h, self.config.eps
        if u < eps:
            val = ((t * h) ** eps) * ((u / eps) ** (eps / (1 - eps)))
        else:
            val = (1 - eps) * ((t * h) ** eps) / (1 - u)
        return self.tau(val)

    def inverse_lambda(self, x: float) -> float:
        h, eps = self.config.h, self.config.eps
        return (x * (1 - eps) * (h**eps)) ** (1 / (1 - eps))

    def compute_small_jump_variance(self, x_prev: float, t_prev: float, t_curr: float) -> float:
        h, eps, alpha = self.config.h, self.config.eps, self.config.alpha

        def inner_int(s: float) -> float:
            if s == 0:
                return 0
            return self.tau((s * h) ** eps) ** (2 - alpha) / (2 - alpha)

        return 2 * (np.cos(x_prev) ** 2) * (t_curr - t_prev) * inner_int(t_prev)
