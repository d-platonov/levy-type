from functools import cached_property

import numpy as np
import pandas as pd
from numba import njit

from levy_type.ar_process import ARProcess
from levy_type.base_process import BaseProcess, JumpSign
from levy_type.dc_process import DCProcess
from levy_type.process_simulator import Path
from levy_type.simulation_config import SimulationConfigAR, SimulationConfigDC


@njit
def simulate_path_ar_numba(
    times: np.ndarray,
    jumps: np.ndarray,
    normal_draws: np.ndarray,
    x0: float,
    eps: float,
    alpha: float,
) -> np.ndarray:
    N = times.shape[0]
    values = np.empty(N, dtype=np.float64)
    values[0] = x0
    const = eps ** (2 - alpha) / (2 - alpha)
    for i in range(1, N):
        dt = times[i] - times[i - 1]
        x_prev = values[i - 1]
        s, c = np.sin(x_prev), np.cos(x_prev)
        drift_inc = s * dt
        large_jump_inc = c * jumps[i]
        var_small = 2.0 * c * c * dt * const
        small_jump_inc = np.sqrt(var_small) * normal_draws[i]
        values[i] = x_prev + drift_inc + large_jump_inc + small_jump_inc
    return values


@njit
def tau_dc(t: float, alpha: float) -> float:
    return ((alpha + t) / t) ** (-1.0 / alpha)


@njit
def simulate_path_dc_numba(
    times: np.ndarray,
    jumps: np.ndarray,
    normal_draws: np.ndarray,
    x0: float,
    h: float,
    eps: float,
    alpha: float,
) -> np.ndarray:
    N = times.shape[0]
    values = np.empty(N, dtype=np.float64)
    values[0] = x0
    for i in range(1, N):
        dt = times[i] - times[i - 1]
        x_prev = values[i - 1]
        drift_inc = np.sin(x_prev) * dt
        large_jump_inc = np.cos(x_prev) * jumps[i]
        inner_val = (
            0.0 if times[i - 1] == 0.0 else (tau_dc((times[i - 1] * h) ** eps, alpha) ** (2 - alpha)) / (2 - alpha)
        )
        var_small = 2.0 * (np.cos(x_prev) ** 2) * dt * inner_val
        small_jump_inc = np.sqrt(var_small) * normal_draws[i]
        values[i] = x_prev + drift_inc + large_jump_inc + small_jump_inc
    return values


@njit
def find_max_diff_numba(
    times_bench: np.ndarray,
    values_bench: np.ndarray,
    times_coarse: np.ndarray,
    values_coarse: np.ndarray,
) -> float:
    i, j, max_diff = 0, 0, 0.0
    n_b, n_c = times_bench.shape[0], times_coarse.shape[0]
    last_b, last_c = values_bench[0], values_coarse[0]
    while i < n_b and j < n_c:
        if times_bench[i] < times_coarse[j]:
            last_b = values_bench[i]
            max_diff = max(max_diff, abs(last_b - last_c))
            i += 1
        elif times_bench[i] > times_coarse[j]:
            last_c = values_coarse[j]
            max_diff = max(max_diff, abs(last_b - last_c))
            j += 1
        else:
            last_b, last_c = values_bench[i], values_coarse[j]
            max_diff = max(max_diff, abs(last_b - last_c))
            i += 1
            j += 1
    while i < n_b:
        last_b = values_bench[i]
        max_diff = max(max_diff, abs(last_b - last_c))
        i += 1
    while j < n_c:
        last_c = values_coarse[j]
        max_diff = max(max_diff, abs(last_b - last_c))
        j += 1
    return max_diff


class SimulatorNumba:
    def __init__(
        self,
        process: BaseProcess,
        k_values: list[int] | None = None,
        approximate_small_jumps: bool = True,
    ) -> None:
        self.process = process
        self.k_values = sorted(k_values) if k_values else []
        self.approximate_small_jumps = approximate_small_jumps

    @cached_property
    def regular_times(self) -> np.ndarray:
        return np.linspace(0.0, self.process.config.total_time, self.process.config.num_steps)

    def generate_grid(self) -> tuple[np.ndarray, np.ndarray]:
        times = self.regular_times
        jump_times: list[float] = []
        jump_sizes: list[float] = []
        for sign in JumpSign:
            jt = self.process.generate_large_jump_times(self.process.config.total_time)
            js = [self.process.sample_large_jump(t, sign) for t in jt]
            jump_times.extend(jt)
            jump_sizes.extend(js)
        all_times = np.concatenate([times, np.array(jump_times)])
        all_times.sort()
        jump_values = np.zeros_like(all_times)
        jump_values[np.searchsorted(all_times, jump_times)] = jump_sizes
        return all_times, jump_values

    def generate_random_draws(self, num: int) -> np.ndarray:
        return self.process.rng.standard_normal(num)

    def simulate_path(self, times: np.ndarray, jumps: np.ndarray, normal_draws: np.ndarray) -> Path:
        config = self.process.config
        if isinstance(config, SimulationConfigAR):
            values = simulate_path_ar_numba(times, jumps, normal_draws, config.x_0, config.eps, config.alpha)
        elif isinstance(config, SimulationConfigDC):
            values = simulate_path_dc_numba(times, jumps, normal_draws, config.x_0, config.h, config.eps, config.alpha)
        else:
            raise ValueError("Unsupported config.")
        return Path(times, values)

    def get_coarse_grid(
        self, k: int, times_bench: np.ndarray, jumps_bench: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        coarse_regular = self.regular_times[::k]
        jump_times = times_bench[jumps_bench != 0]
        coarse_times = np.union1d(coarse_regular, jump_times)
        coarse_jumps = np.zeros_like(coarse_times)
        indices = np.searchsorted(coarse_times, times_bench[jumps_bench != 0])
        coarse_jumps[indices] = jumps_bench[jumps_bench != 0]
        return coarse_times, coarse_jumps

    def simulate_synchronized_paths(self) -> dict[int, Path]:
        times_bench, jumps_bench = self.generate_grid()
        normal_draws_bench = self.generate_random_draws(len(times_bench))
        benchmark = self.simulate_path(times_bench, jumps_bench, normal_draws_bench)
        paths: dict[int, Path] = {1: benchmark}
        for k in self.k_values:
            coarse_times, coarse_jumps = self.get_coarse_grid(k, times_bench, jumps_bench)
            indices = np.searchsorted(times_bench, coarse_times)
            normal_draws_coarse = normal_draws_bench[indices]
            paths[k] = self.simulate_path(coarse_times, coarse_jumps, normal_draws_coarse)
        return paths

    def simulate_multiple_paths(self, n: int) -> list[dict[int, Path]]:
        return [self.simulate_synchronized_paths() for _ in range(n)]


class ErrorEstimator:
    def __init__(self, simulator: SimulatorNumba) -> None:
        self.simulator = simulator

    def estimate_error(self, k_values: list[int], p_values: list[int], n_paths: int) -> pd.DataFrame:
        simulations = self.simulator.simulate_multiple_paths(n_paths)
        diffs = np.zeros((len(k_values), len(simulations)))
        for i, k in enumerate(k_values):
            for j, sim_paths in enumerate(simulations):
                bench = sim_paths[1]
                coarse = sim_paths[k]
                diffs[i, j] = find_max_diff_numba(bench.times, bench.values, coarse.times, coarse.values)
        results = pd.DataFrame(index=k_values, columns=p_values)
        for p in p_values:
            results.loc[:, p] = np.mean(diffs**p, axis=1) ** (1 / p)
        return results


def compare_ar_and_dc(
    n_paths: int, n_batches: int, seed: int, alpha: float, eps_ar: float, eps_dc: float, h_dc: float
) -> None:
    print("")
    print(f"Simulations for alpha = {alpha}")

    print(f"Using AR process (eps = {eps_ar})")
    config_ar = SimulationConfigAR(
        total_time=1.0,
        num_steps=2**17,
        x_0=0.0,
        alpha=alpha,
        random_seed=seed,
        eps=eps_ar,
    )
    process_ar = ARProcess(config=config_ar)
    simulator_ar = SimulatorNumba(process=process_ar, k_values=[2, 4, 8, 16, 32, 64, 128, 256])
    estimator_ar = ErrorEstimator(simulator=simulator_ar)
    result_ar = pd.DataFrame(columns=[2, 4, 6, 8, 10], index=[2, 4, 8, 16, 32, 64, 128, 256], data=np.zeros((8, 5)))
    for _ in range(n_batches):
        result_ar += estimator_ar.estimate_error(
            k_values=[2, 4, 8, 16, 32, 64, 128, 256],
            p_values=[2, 4, 6, 8, 10],
            n_paths=n_paths,
        )
    print(result_ar / n_batches)

    print(f"Using DC process (eps = {eps_dc}, h = {h_dc})")
    config_dc = SimulationConfigDC(
        total_time=1.0,
        num_steps=2**17,
        x_0=0.0,
        alpha=alpha,
        random_seed=seed,
        h=h_dc,
        eps=eps_dc,
    )
    process_dc = DCProcess(config=config_dc)
    simulator_dc = SimulatorNumba(process=process_dc, k_values=[2, 4, 8, 16, 32, 64, 128, 256])
    estimator_dc = ErrorEstimator(simulator=simulator_dc)
    result_dc = pd.DataFrame(columns=[2, 4, 6, 8, 10], index=[2, 4, 8, 16, 32, 64, 128, 256], data=np.zeros((8, 5)))
    for _ in range(n_batches):
        result_dc += estimator_dc.estimate_error(
            k_values=[2, 4, 8, 16, 32, 64, 128, 256],
            p_values=[2, 4, 6, 8, 10],
            n_paths=n_paths,
        )
    print(result_dc / n_batches)


if __name__ == "__main__":
    seed = 2025
    n_paths = 10  # 1_000
    print("AR variance == DC variance.")

    compare_ar_and_dc(n_paths=n_paths, n_batches=100, seed=seed, alpha=0.5, eps_ar=0.01, eps_dc=0.1, h_dc=6.810e-13)
    compare_ar_and_dc(n_paths=n_paths, n_batches=100, seed=seed, alpha=1.0, eps_ar=0.01, eps_dc=0.1, h_dc=2.877e-20)
    compare_ar_and_dc(n_paths=n_paths, n_batches=100, seed=seed, alpha=1.5, eps_ar=0.01, eps_dc=0.1, h_dc=1.575e-28)
