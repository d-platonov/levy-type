from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from levy_type.simulation_config import SimulationConfig


class JumpSign(IntEnum):
    NEGATIVE = -1
    POSITIVE = +1


@dataclass
class BaseProcess(ABC):
    config: SimulationConfig

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.config.random_seed)

    @staticmethod
    @abstractmethod
    def drift_coefficient(t: float, x: float) -> float:
        pass

    @staticmethod
    @abstractmethod
    def diffusion_coefficient(t: float, x: float) -> float:
        pass

    @staticmethod
    @abstractmethod
    def jump_coefficient(t: float, x: float, z: float) -> float:
        pass

    @abstractmethod
    def inverse_large_jump_cdf(self, u: float, t: float | None) -> float:
        """Inverse CDF for large jumps."""
        pass

    @abstractmethod
    def inverse_lambda(self, x: float) -> float:
        """Inverse of large-jump intensity."""
        pass

    @abstractmethod
    def compute_small_jump_variance(self, x_prev: float, t_prev: float, t_curr: float) -> float:
        """Variance of small jumps [t_prev, t_curr]."""
        pass

    def sample_large_jump(self, t: float, sign: JumpSign) -> float:
        """Sample a large jump at time t."""
        u = self.rng.uniform()
        return float(sign) * self.inverse_large_jump_cdf(u, t)

    def generate_large_jump_times(self, t: float) -> list[float]:
        """Return large jump times up to time t."""
        times = []
        exp_sum = self.rng.exponential()
        while (jump_time := self.inverse_lambda(exp_sum)) < t:
            times.append(jump_time)
            exp_sum += self.rng.exponential()
        return times
