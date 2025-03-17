# DC VS AR Approaches for Jump-Driven SDEs

This repository compares two numerical methods for approximating jump-driven SDEs:
- **Dynamic Cutting (DC) approach**
- **Asmussen-Rosinski (AR) approach**

## Process Description

$$
\begin{aligned}
dX_t &= \sin(X_t)\,dt + \int_{\mathbb{R}} \cos(X_{t-})\,z\,\widetilde{N}(dt,dz), \\
X_0 &= 0,
\end{aligned}
$$

with Lévy measure

$$
\nu(dz) = \mathbf{1}_{\{|z| \leq 1\}} \frac{dz}{|z|^{1+\alpha}}.
$$

## Simulation Setup

- **Benchmark Path:** A path with $2^{17}$ points is generated.
- **Coarse Grids:** For coarser grids ranging from $2^9$ to $2^{16}$ points, we compute the strong error in the $L^p(\Omega)$-norm for $p = 2, 4, 6, 8, 10$:

$$
\left\|\sup_{0\le t\le 1} \left|X_t^{\mathrm{benchmark}} - X_t^{\mathrm{coarse}}\right|\right\|_{L^p(\Omega)}.
$$

## Repository Structure

- **`simulation_config.py`** – Simulation configuration parameters.
- **`ar_process.py` / `dc_process.py`** – Implementations of the AR and DC processes.
- **`process_simulator.py`** – Simulation of benchmark and coarse paths.
- **`error_estimator.py`** – Estimation of strong errors.
- **`main.py`** – Main script to run simulations and comparisons.
- **`run_numba.py`** – Fast script to run simulations and comparisons (in batches).
