from dataclasses import dataclass
from functools import cached_property

import numpy as np
from levy_type.base_process import BaseProcess, JumpSign


@dataclass(slots=True)
class Path:
    times: np.ndarray
    values: np.ndarray


class ProcessSimulator:
    def __init__(self, process: BaseProcess, k_values: list[int] | None = None, approximate_small_jumps: bool = True):
        self.process = process
        self.k_values = sorted(k_values) if k_values else []
        self.approximate_small_jumps = approximate_small_jumps

    @cached_property
    def regular_times(self) -> np.ndarray:
        return np.linspace(0.0, self.process.config.total_time, self.process.config.num_steps)

    def generate_grid_2d(self) -> tuple[np.ndarray, np.ndarray]:
        times = self.regular_times
        total_time = self.process.config.total_time
        jump_times, jump_sizes = [], []

        for sign in JumpSign:
            jt = self.process.generate_large_jump_times(total_time)
            js = [self.process.sample_large_jump(t, sign) for t in jt]
            jump_times.extend(jt)
            jump_sizes.extend(js)

        all_times = np.sort(np.concatenate([times, jump_times]))
        jump_values = np.zeros_like(all_times)
        jump_values[np.searchsorted(all_times, jump_times)] = jump_sizes

        return all_times, jump_values

    def generate_random_draws(self, num_steps: int) -> np.ndarray:
        return self.process.rng.standard_normal(num_steps)

    def simulate_path(self, times: np.ndarray, jumps: np.ndarray, normal_draws: np.ndarray) -> Path:
        # TODO: add diffusion part with its own normal_draws
        values = np.empty_like(times, dtype=np.float64)
        values[0] = self.process.config.x_0
        for i in range(1, len(times)):
            t_prev, t_curr = times[i - 1], times[i]
            x_prev = values[i - 1]
            dt = t_curr - t_prev
            drift = self.process.drift_coefficient(t_prev, x_prev) * dt
            jump = self.process.jump_coefficient(t_prev, x_prev, jumps[i]) if jumps[i] else 0.0
            small_jump = (
                np.sqrt(self.process.compute_small_jump_variance(x_prev, t_prev, t_curr)) * normal_draws[i]
                if self.approximate_small_jumps
                else 0.0
            )
            values[i] = x_prev + drift + jump + small_jump
        return Path(times, values)

    def _get_coarse_times(self, k: int, times_benchmark: np.ndarray, jumps_benchmark: np.ndarray) -> np.ndarray:
        coarse_regular = self.regular_times[::k]
        jump_times = times_benchmark[jumps_benchmark != 0]
        return np.union1d(coarse_regular, jump_times)

    @staticmethod
    def _get_coarse_normal_draws(
        coarse_times: np.ndarray, times_benchmark: np.ndarray, normal_draws_benchmark: np.ndarray
    ) -> np.ndarray:
        indices = np.searchsorted(times_benchmark, coarse_times)
        return normal_draws_benchmark[indices]

    @staticmethod
    def _get_coarse_jumps(
        coarse_times: np.ndarray, times_benchmark: np.ndarray, jumps_benchmark: np.ndarray
    ) -> np.ndarray:
        mask = jumps_benchmark != 0
        indices = np.searchsorted(coarse_times, times_benchmark[mask])
        jumps_coarse = np.zeros_like(coarse_times)
        jumps_coarse[indices] = jumps_benchmark[mask]
        return jumps_coarse

    def simulate_synchronized_paths(self) -> dict[int, Path]:
        times_benchmark, jumps_benchmark = self.generate_grid_2d()
        normal_draws_benchmark = self.generate_random_draws(len(times_benchmark))
        benchmark_path = self.simulate_path(times_benchmark, jumps_benchmark, normal_draws_benchmark)

        paths_dict = {1: benchmark_path}
        for k in self.k_values:
            coarse_times = self._get_coarse_times(k, times_benchmark, jumps_benchmark)
            coarse_jumps = self._get_coarse_jumps(coarse_times, times_benchmark, jumps_benchmark)
            normal_draws_coarse = self._get_coarse_normal_draws(coarse_times, times_benchmark, normal_draws_benchmark)
            paths_dict[k] = self.simulate_path(coarse_times, coarse_jumps, normal_draws_coarse)

        return paths_dict

    def simulate_paths(self, n: int) -> list[dict[int, Path]]:
        return [self.simulate_synchronized_paths() for _ in range(n)]
