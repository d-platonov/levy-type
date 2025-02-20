from dataclasses import dataclass

import numpy as np

from levy_type.base_process import BaseProcess
from levy_type.simulation_config import SimulationConfigAR


@dataclass
class ARProcess(BaseProcess):
    config: SimulationConfigAR

    @staticmethod
    def drift_coefficient(t: float, x: float) -> float:
        return np.sin(x)

    @staticmethod
    def diffusion_coefficient(t: float, x: float) -> float:
        return 0

    @staticmethod
    def jump_coefficient(t: float, x: float, z: float) -> float:
        return np.cos(x) * z

    def inverse_large_jump_cdf(self, u: float, t: float) -> float:
        eps, alpha = self.config.eps, self.config.alpha
        return (-u * eps ** (-alpha) + eps ** (-alpha) + u) ** (-1 / alpha)

    def inverse_lambda(self, x: float) -> float:
        eps, alpha = self.config.eps, self.config.alpha
        return x * alpha / (eps ** (-alpha) - 1)

    def compute_small_jump_variance(self, x_prev: float, t_prev: float, t_curr: float) -> float:
        eps, alpha = self.config.eps, self.config.alpha
        return 2 * (np.cos(x_prev) ** 2) * (t_curr - t_prev) * eps ** (2 - alpha) / (2 - alpha)
