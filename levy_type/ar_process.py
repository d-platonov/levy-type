from dataclasses import dataclass

from levy_type.base_process import BaseProcess
from levy_type.simulation_config import ARSimulationConfig


@dataclass
class ARProcess(BaseProcess):
    config: ARSimulationConfig

    @staticmethod
    def drift_coefficient(t: float, x: float) -> float:
        return 0  # np.sin(x)

    @staticmethod
    def diffusion_coefficient(t: float, x: float) -> float:
        return 0

    @staticmethod
    def jump_coefficient(t: float, x: float, z: float) -> float:
        return z

    def large_jump_cdf_inverse(self, u: float, t: float | None = None) -> float:
        eps, alpha = self.config.eps, self.config.alpha
        return (-u * eps ** (-alpha) + eps ** (-alpha) + u) ** (-1 / alpha)

    def large_jump_lambda_inverse(self, x: float) -> float:
        eps, alpha = self.config.eps, self.config.alpha
        return x * alpha / (eps ** (-alpha) - 1)

    def small_jump_variance(self, x_prev: float, t_prev: float, t_curr: float) -> float:
        eps, alpha = self.config.eps, self.config.alpha
        return 2 * (t_curr - t_prev) * eps ** (2 - alpha) / (2 - alpha)
