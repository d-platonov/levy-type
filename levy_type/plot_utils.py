from matplotlib import pyplot as plt

from levy_type.coupled_process_simulator import CoupledPaths
from levy_type.process_simulator import Path


def plot_paths(
    paths: Path | list[Path],
    title: str = "Simulated Process Paths",
    xlabel: str = "Time",
    ylabel: str = "Value",
) -> None:
    if not isinstance(paths, list):
        paths = [paths]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    for i, path in enumerate(paths):
        ax.plot(path.times, path.values, drawstyle='steps-post', label=f'Path {i + 1}', alpha=0.8, linewidth=1.5)

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


def plot_coupled_paths(
    coupled_paths: CoupledPaths,
    title: str = "Coupled Fine and Coarse Path Simulation",
) -> None:
    fine_path = coupled_paths.fine_path
    coarse_path = coupled_paths.coarse_path

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(
        fine_path.times,
        fine_path.values,
        drawstyle='steps-post',
        label=f'Fine Path (N={len(fine_path.values) - 1})',
        color='cornflowerblue',
        linewidth=1.0,
        alpha=0.8,
    )

    ax.plot(
        coarse_path.times,
        coarse_path.values,
        drawstyle='steps-post',
        label=f'Coarse Path (N={len(coarse_path.values) - 1})',
        color='crimson',
        linewidth=2.0,
        alpha=0.9,
    )

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()
