from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from levy_type.base_process import BaseProcess, JumpSign
from levy_type.process_factory import ProcessFactory
from levy_type.process_simulator import Path
from levy_type.simulation_config import (
    ARSimulationConfig,
    DCSimulationConfig,
    SimulationConfig,
)


@dataclass(slots=True)
class CoupledPaths:
    fine_path: Path
    coarse_path: Path


@dataclass(slots=True)
class PathComponents:
    fine_path: Path
    jump_times: np.ndarray
    jump_sizes: np.ndarray
    gaussian_draws: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    diffusion_variances: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))


# TODO: Add diffusion part of the SDE, currently it is not needed since in the example it is equal to 0, so it's omitted
class CoupledSimulator:
    """Generates coupled fine and coarse paths for strong-error estimation."""

    def __init__(
        self,
        fine_config: SimulationConfig,
        coarse_config: SimulationConfig,
        approximate_small_jumps: bool = True,
    ):
        if fine_config.T != coarse_config.T:
            raise ValueError("Time horizon T must match.")
        if fine_config.N < coarse_config.N:
            raise ValueError("Fine N must be >= coarse N.")
        if fine_config.N % coarse_config.N != 0:
            raise ValueError("Fine N must be a multiple of coarse N.")

        self.fine_config = fine_config
        self.coarse_config = coarse_config
        self.approximate_small_jumps = approximate_small_jumps

        seed = getattr(fine_config, "random_seed", 42)
        self.rng = np.random.default_rng(seed)

        self.process_fine = ProcessFactory.create_process(fine_config)
        self.process_coarse = ProcessFactory.create_process(coarse_config)

        self.process_fine.rng = self.rng
        self.process_coarse.rng = self.rng

    @staticmethod
    def _sample_all_jumps(process: BaseProcess) -> tuple[np.ndarray, np.ndarray]:
        """Samples and sorts all large jumps for a given process."""
        all_times, all_sizes = [], []
        for sign in JumpSign:
            times = process.sample_large_jump_times(process.config.T)
            if times:
                sizes = [process.sample_large_jump_sizes(t, sign) for t in times]
                all_times.extend(times)
                all_sizes.extend(sizes)

        if not all_times:
            return np.empty(0, dtype=float), np.empty(0, dtype=float)

        # Sort jumps chronologically
        order = np.argsort(all_times)
        return np.array(all_times)[order], np.array(all_sizes)[order]

    @staticmethod
    def _place_jumps_on_grid(time_grid, jump_times, jump_sizes) -> np.ndarray:
        """Aggregates jump sizes onto the corresponding time grid indices."""
        jumps_on_grid = np.zeros_like(time_grid, dtype=float)
        if jump_times.size > 0:
            indices = np.searchsorted(time_grid, jump_times, side="left")
            np.add.at(jumps_on_grid, indices, jump_sizes)
        return jumps_on_grid

    def _simulate_fine_and_get_components(self) -> PathComponents:
        """Simulates the fine path and extracts its random components."""
        process, config = self.process_fine, self.fine_config

        jump_times, jump_sizes = self._sample_all_jumps(process)

        regular_times = np.linspace(0.0, config.T, config.N + 1, dtype=float)
        t_fine = np.union1d(regular_times, jump_times)
        jumps_on_grid = self._place_jumps_on_grid(t_fine, jump_times, jump_sizes)

        values = np.empty_like(t_fine)
        values[0] = config.x_0

        num_intervals = len(t_fine) - 1
        gaussian_draws = np.zeros(num_intervals)
        diffusion_variances = np.zeros(num_intervals)

        for i in range(1, len(t_fine)):
            t_prev, t_curr = t_fine[i - 1], t_fine[i]
            x_prev = values[i - 1]
            dt = t_curr - t_prev

            x_next = x_prev + process.drift_coefficient(t_prev, x_prev) * dt
            x_next += process.jump_coefficient(t_prev, x_prev, jumps_on_grid[i])

            if self.approximate_small_jumps and dt > 0.0:
                variance = process.small_jump_variance(x_prev, t_prev, t_curr)
                if variance > 0.0:
                    z_draw = self.rng.standard_normal()
                    x_next += np.sqrt(variance) * z_draw
                    gaussian_draws[i - 1] = z_draw
                    diffusion_variances[i - 1] = variance
            values[i] = x_next

        return PathComponents(
            fine_path=Path(times=t_fine, values=values),
            jump_times=jump_times,
            jump_sizes=jump_sizes,
            gaussian_draws=gaussian_draws,
            diffusion_variances=diffusion_variances,
        )

    def _coarse_threshold(self, t: float) -> float:
        """Returns the large-jump threshold for the coarse process at a given time."""
        process, config = self.process_coarse, self.coarse_config
        if isinstance(config, ARSimulationConfig):
            return config.eps
        if isinstance(config, DCSimulationConfig):
            return process.tau((t * config.h) ** config.eps)
        raise NotImplementedError("Unknown process type for coarse threshold.")

    def _thin_jumps_for_coarse(self, jump_times, jump_sizes) -> tuple[np.ndarray, np.ndarray]:
        """Filters fine jumps, keeping only those large enough for the coarse scheme."""
        if jump_times.size == 0:
            return jump_times, jump_sizes

        keep_mask = [abs(z) >= self._coarse_threshold(t) for t, z in zip(jump_times, jump_sizes)]
        return jump_times[keep_mask], jump_sizes[keep_mask]

    def _simulate_coarse_from_components(self, components: PathComponents) -> Path:
        """Builds the coarse path using the components from the fine path."""
        process, config = self.process_coarse, self.coarse_config

        level_ratio = self.fine_config.N // config.N
        fine_regular_times = np.linspace(0.0, self.fine_config.T, self.fine_config.N + 1)
        coarse_regular_times = fine_regular_times[::level_ratio]

        thinned_times, thinned_sizes = self._thin_jumps_for_coarse(components.jump_times, components.jump_sizes)
        t_coarse = np.union1d(coarse_regular_times, thinned_times)
        jumps_on_grid = self._place_jumps_on_grid(t_coarse, thinned_times, thinned_sizes)

        map_coarse_to_fine_idx = np.searchsorted(components.fine_path.times, t_coarse, side="left")
        fine_gaussian_draws = components.gaussian_draws

        values = np.empty_like(t_coarse)
        values[0] = config.x_0

        for i in range(1, len(t_coarse)):
            t_prev, t_curr = t_coarse[i - 1], t_coarse[i]
            x_prev = values[i - 1]
            dt = t_curr - t_prev

            x_next = x_prev + process.drift_coefficient(t_prev, x_prev) * dt
            x_next += process.jump_coefficient(t_prev, x_prev, jumps_on_grid[i])

            if self.approximate_small_jumps and dt > 0.0:
                v_coarse = process.small_jump_variance(x_prev, t_prev, t_curr)
                if v_coarse > 0.0:
                    start_idx_f = map_coarse_to_fine_idx[i - 1]
                    end_idx_f = map_coarse_to_fine_idx[i]

                    if end_idx_f > start_idx_f:
                        z_coupled = fine_gaussian_draws[start_idx_f]
                        x_next += np.sqrt(v_coarse) * z_coupled
                    else:
                        x_next += np.sqrt(v_coarse) * self.rng.standard_normal()
            values[i] = x_next

        return Path(times=t_coarse, values=values)

    def simulate(self) -> CoupledPaths:
        """Simulates a single pair of coupled fine and coarse paths."""
        components = self._simulate_fine_and_get_components()
        coarse_path = self._simulate_coarse_from_components(components)
        return CoupledPaths(fine_path=components.fine_path, coarse_path=coarse_path)

    def simulate_many(self, num_paths: int) -> list[CoupledPaths]:
        """Simulates multiple pairs of coupled paths."""
        return [self.simulate() for _ in range(num_paths)]
