from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from levy_type.simulation_config import SimulationConfig


class JumpSign(IntEnum):
    NEGATIVE = -1
    POSITIVE = +1


@dataclass
class BaseProcess(ABC):
    config: SimulationConfig
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.config.random_seed)

    @staticmethod
    @abstractmethod
    def drift_coefficient(t: float, x: float) -> float:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def diffusion_coefficient(t: float, x: float) -> float:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def jump_coefficient(t: float, x: float, z: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def large_jump_lambda_inverse(self, x: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def large_jump_cdf_inverse(self, u: float, t: float | None = None) -> float:
        raise NotImplementedError

    @abstractmethod
    def small_jump_variance(self, x_prev: float, t_prev: float, t_curr: float) -> float:
        raise NotImplementedError

    def sample_large_jump_times(self, t: float) -> list[float]:
        times = []
        exp_sum = self.rng.exponential()
        while (jump_time := self.large_jump_lambda_inverse(exp_sum)) < t:
            times.append(jump_time)
            exp_sum += self.rng.exponential()
        return times

    def sample_large_jump_sizes(self, t: float, sign: JumpSign) -> float:
        u = self.rng.uniform()
        return float(sign) * self.large_jump_cdf_inverse(u, t)
