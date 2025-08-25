from levy_type.coupled_process_simulator import CoupledSimulator
from levy_type.error_estimator import ErrorEstimator
from levy_type.plot_utils import plot_coupled_paths, plot_paths
from levy_type.process_simulator import ProcessSimulator
from levy_type.simulation_config import ARSimulationConfig, DCSimulationConfig


def main():
    # Plot some trajectories
    config_dc = DCSimulationConfig(
        T=1.0,
        N=100,
        h=3e-5,
        eps=0.25,
        x_0=0.0,
        alpha=0.5,
        random_seed=12,
    )
    simulator_dc = ProcessSimulator(config=config_dc)
    paths_dc = simulator_dc.simulate_many(n=100)
    plot_paths(paths_dc)

    # Plot coupled paths
    fine_config = ARSimulationConfig(
        T=1.0,
        N=16,
        eps=0.1,
        x_0=0.0,
        alpha=0.5,
        random_seed=12,
    )
    coarse_config = ARSimulationConfig(
        T=1.0,
        N=4,
        eps=0.1,
        x_0=0.0,
        alpha=0.5,
        random_seed=12,
    )
    simulator = CoupledSimulator(
        fine_config=fine_config,
        coarse_config=coarse_config,
    )
    coupled_paths = simulator.simulate()
    plot_coupled_paths(coupled_paths)

    # Estimate strong errors
    print("\n--- Strong Error Estimation Results ---")
    fine_config_ar = ARSimulationConfig(
        T=1.0,
        N=2**14,
        eps=0.01183550570,
        x_0=0.0,
        alpha=0.5,
        random_seed=42,
    )
    fine_config_dc = DCSimulationConfig(
        T=1.0,
        N=2**14,
        eps=0.25,
        h=3e-5,
        x_0=0.0,
        alpha=0.5,
        random_seed=42,
    )

    print("AR:")
    estimator = ErrorEstimator(fine_config=fine_config_ar, level_ratios=[2, 4, 8], p_values=[2], n_paths=100)
    error_df = estimator.estimate_strong_error()
    print(error_df)

    print("DC:")
    estimator = ErrorEstimator(fine_config=fine_config_dc, level_ratios=[2, 4, 8], p_values=[2], n_paths=100)
    error_df = estimator.estimate_strong_error()
    print(error_df)


if __name__ == "__main__":
    main()
