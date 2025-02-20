from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from levy_type.base_process import BaseProcess
from levy_type.process_factory import ProcessFactory
from levy_type.process_simulator import Path, ProcessSimulator
from levy_type.simulation_config import SimulationConfig


@dataclass
class ErrorEstimator:
    config: SimulationConfig
    k_values: list[int]
    p_values: list[int]
    n_paths: int
    process: BaseProcess = field(init=False)
    simulator: ProcessSimulator = field(init=False)

    def __post_init__(self) -> None:
        self.process = ProcessFactory.create_process(config=self.config)
        self.simulator = ProcessSimulator(process=self.process, k_values=self.k_values)

    @staticmethod
    def _find_max_diff_between_two_paths(path_benchmark: Path, path_coarse: Path) -> float:
        """Max abs diff between paths."""
        tb, vb = path_benchmark.times, path_benchmark.values
        tc, vc = path_coarse.times, path_coarse.values
        idx_c = np.searchsorted(tc, tb, side="right") - 1
        idx_c = np.clip(idx_c, 0, len(tc) - 1)
        return np.max(np.abs(vb - vc[idx_c]))

    def estimate_error(self) -> pd.DataFrame:
        """Estimate strong error."""
        p_arr = np.array(self.p_values, dtype=np.float64)
        sums = np.zeros((len(self.k_values), p_arr.size), dtype=np.float64)

        for _ in range(self.n_paths):
            sim = self.simulator.simulate_synchronized_paths()
            bench = sim[1]
            for i, k in enumerate(self.k_values):
                diff = self._find_max_diff_between_two_paths(bench, sim[k])
                sums[i, :] += diff**p_arr

        means = sums / self.n_paths
        errors = means ** (1 / p_arr)
        return pd.DataFrame(errors, index=self.k_values, columns=self.p_values)
