from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from attrs import asdict

from levy_type.coupled_process_simulator import CoupledSimulator
from levy_type.process_simulator import Path
from levy_type.simulation_config import SimulationConfig


@dataclass
class ErrorEstimator:
    """
    Estimates the strong error of a numerical scheme by comparing coarse paths
    against a fine benchmark path.
    """
    fine_config: SimulationConfig
    level_ratios: list[int]
    p_values: list[int]
    n_paths: int

    @staticmethod
    def _find_max_diff_between_two_paths(path_fine: Path, path_coarse: Path) -> float:
        """Calculates the maximum absolute difference between two paths over time."""
        tb, vb = path_fine.times, path_fine.values
        tc, vc = path_coarse.times, path_coarse.values

        # For each time in the fine path, find the index of the corresponding
        # time in the coarse path (the latest time that is not after it).
        idx_c = np.searchsorted(tc, tb, side="right") - 1
        idx_c = np.clip(idx_c, 0, len(tc) - 1)

        return np.max(np.abs(vb - vc[idx_c]))

    def _create_coarse_config(self, level_ratio: int) -> SimulationConfig:
        """Creates a coarse configuration from the fine one using a level ratio."""
        if self.fine_config.N % level_ratio != 0:
            raise ValueError(f"Fine N ({self.fine_config.N}) must be a multiple of "
                             f"level_ratio ({level_ratio}).")

        coarse_n = self.fine_config.N // level_ratio

        params = asdict(self.fine_config)
        params['N'] = coarse_n

        return type(self.fine_config)(**params)

    def estimate_strong_error(self) -> pd.DataFrame:
        """Runs the Monte Carlo simulation to estimate the strong Lp error."""
        results = []
        p_arr = np.array(self.p_values, dtype=np.float64)

        for L in self.level_ratios:
            print(f"Estimating error for level ratio L={L}...")

            coarse_config = self._create_coarse_config(L)

            simulator = CoupledSimulator(
                fine_config=self.fine_config,
                coarse_config=coarse_config,
                approximate_small_jumps=True,
            )

            coupled_paths_list = simulator.simulate_many(self.n_paths)

            sums = np.zeros_like(p_arr)
            for coupled_paths in coupled_paths_list:
                diff = self._find_max_diff_between_two_paths(
                    coupled_paths.fine_path, coupled_paths.coarse_path
                )
                sums += diff ** p_arr

            mean_of_powers = sums / self.n_paths
            errors = mean_of_powers ** (1 / p_arr)
            results.append(errors)

        return pd.DataFrame(
            results,
            index=self.level_ratios,
            columns=self.p_values
        )