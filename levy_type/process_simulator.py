from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from levy_type.base_process import JumpSign
from levy_type.process_factory import ProcessFactory
from levy_type.simulation_config import SimulationConfig


@dataclass(slots=True)
class Path:
    times: np.ndarray
    values: np.ndarray


class ProcessSimulator:
    def __init__(self, config: SimulationConfig, approximate_small_jumps: bool = True):
        self.config = config
        self.process = ProcessFactory.create_process(config)
        self.approximate_small_jumps = approximate_small_jumps

    def simulate(self) -> Path:
        process, config = self.process, self.config

        jump_times_list, jump_sizes_list = [], []
        for sign in JumpSign:
            times = np.asarray(process.sample_large_jump_times(config.T))
            if times.size > 0:
                sizes = np.array([process.sample_large_jump_sizes(t, sign) for t in times])
                jump_times_list.append(times)
                jump_sizes_list.append(sizes)

        if jump_times_list:
            all_jump_times = np.concatenate(jump_times_list)
            all_jump_sizes = np.concatenate(jump_sizes_list)
        else:
            all_jump_times, all_jump_sizes = np.empty(0), np.empty(0)

        regular_times = np.linspace(0.0, config.T, config.N)
        time_grid = np.union1d(regular_times, all_jump_times)

        # Place jumps onto the grid
        jumps = np.zeros_like(time_grid)
        if all_jump_times.size > 0:
            indices = np.searchsorted(time_grid, all_jump_times)
            np.add.at(jumps, indices, all_jump_sizes)

        values = np.empty_like(time_grid)
        values[0] = config.x_0

        for i in range(1, len(time_grid)):
            t_prev, t_curr = time_grid[i - 1], time_grid[i]
            x_prev = values[i - 1]
            dt = t_curr - t_prev

            # Drift
            x_next = x_prev + process.drift_coefficient(t_prev, x_prev) * dt

            # Diffusion
            sigma = process.diffusion_coefficient(t_prev, x_prev)
            if sigma != 0.0:
                z_w = process.rng.standard_normal()
                x_next += sigma * np.sqrt(dt) * z_w

            # Large jumps
            if jumps[i] != 0.0:
                x_next += process.jump_coefficient(t_prev, x_prev, jumps[i])

            # Small jumps
            if self.approximate_small_jumps and dt > 0.0:
                variance = process.small_jump_variance(x_prev, t_prev, t_curr)
                if variance > 0.0:
                    z_s = process.rng.standard_normal()  # independent of z_w
                    x_next += np.sqrt(variance) * z_s

            values[i] = x_next

        return Path(times=time_grid, values=values)

    def simulate_many(self, n: int) -> list[Path]:
        return [self.simulate() for _ in range(n)]
