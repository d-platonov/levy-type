from levy_type.ar_process import ARProcess
from levy_type.base_process import BaseProcess
from levy_type.dc_process import DCProcess
from levy_type.simulation_config import SimulationConfig, SimulationConfigAR, SimulationConfigDC


class ProcessFactory:
    _registry = {
        SimulationConfigDC: DCProcess,
        SimulationConfigAR: ARProcess,
    }

    @staticmethod
    def create_process(config: SimulationConfig) -> BaseProcess:
        """Create a process from config."""
        for cfg_type, proc_cls in ProcessFactory._registry.items():
            if isinstance(config, cfg_type):
                return proc_cls(config=config)
        raise ValueError(f"Unsupported config type: {type(config)}")
