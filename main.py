from levy_type.error_estimator import ErrorEstimator
from levy_type.simulation_config import SimulationConfigAR, SimulationConfigDC


def main() -> None:
    """Run error estimation example."""
    alpha = 0.5
    seed = 2025
    n_paths = 10

    config_ar = SimulationConfigAR(
        total_time=1.0,
        num_steps=int(2**17),
        eps=0.01,
        x_0=0.0,
        alpha=alpha,
        random_seed=seed,
    )
    estimator = ErrorEstimator(
        config=config_ar,
        k_values=[2, 4, 8, 16, 32, 64, 128, 256],
        p_values=[2, 4, 6, 8, 10],
        n_paths=n_paths,
    )
    result_ar = estimator.estimate_error()
    print("\nAR, eps = 0.01")
    print(result_ar)

    config_dc = SimulationConfigDC(
        total_time=1.0,
        num_steps=int(2**17),
        h=6.810e-13,
        eps=0.10,
        x_0=0.0,
        alpha=alpha,
        random_seed=seed,
    )
    estimator = ErrorEstimator(
        config=config_dc,
        k_values=[2, 4, 8, 16, 32, 64, 128, 256],
        p_values=[2, 4, 6, 8, 10],
        n_paths=n_paths,
    )
    result_dc = estimator.estimate_error()
    print("\nDC, eps = 0.10, h = 6.810e-13")
    print(result_dc)


if __name__ == "__main__":
    main()
