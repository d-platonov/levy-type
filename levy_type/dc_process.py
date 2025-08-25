from dataclasses import dataclass

from levy_type.base_process import BaseProcess
from levy_type.simulation_config import DCSimulationConfig


@dataclass
class DCProcess(BaseProcess):
    config: DCSimulationConfig

    @staticmethod
    def drift_coefficient(t: float, x: float) -> float:
        return 0  # np.sin(x)

    @staticmethod
    def diffusion_coefficient(t: float, x: float) -> float:
        return 0

    @staticmethod
    def jump_coefficient(t: float, x: float, z: float) -> float:
        return z

    def tau(self, t: float) -> float:
        alpha = self.config.alpha
        return ((alpha + t) / t) ** (-1 / alpha)

    def large_jump_cdf_inverse(self, u: float, t: float | None = None) -> float:
        h, eps = self.config.h, self.config.eps
        val = ((t * h) ** eps) / (1 - u)
        return self.tau(val)

    def large_jump_lambda_inverse(self, x: float) -> float:
        h, eps = self.config.h, self.config.eps
        return (x * (1 - eps) * (h**eps)) ** (1 / (1 - eps))

    def small_jump_variance(self, x_prev: float, t_prev: float, t_curr: float) -> float:
        h, eps, alpha = self.config.h, self.config.eps, self.config.alpha

        def inner_int(s: float) -> float:
            if s <= 0:
                return 0
            return self.tau((s * h) ** eps) ** (2 - alpha) / (2 - alpha)

        # Average the rate at the start and end of the interval
        value_at_start = inner_int(t_prev)
        value_at_end = inner_int(t_curr)

        # The '2 *' from the rate formula cancels with the '/ 2' from averaging
        return (t_curr - t_prev) * (value_at_start + value_at_end)
