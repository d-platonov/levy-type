from attrs import define, field, validators
from attrs.validators import instance_of


@define(kw_only=True)
class SimulationConfig:
    """
    Base configuration for simulating a Lévy-type process.

    Attributes:
        total_time (float): Total simulation time horizon (T); must be > 0.
        num_steps (int): Number of discrete time steps (N); must be > 0.
        x_0 (float): Initial value of the process at t=0.
        alpha (float): Parameter in the Lévy density; must be in (0, 2).
        random_seed (int): Seed for the random number generator.
    """

    total_time: float = field(validator=validators.and_(instance_of(float), validators.gt(0)))
    num_steps: int = field(validator=validators.and_(instance_of(int), validators.gt(0)))
    x_0: float = field(validator=instance_of(float))
    alpha: float = field(validator=validators.and_(instance_of(float), validators.gt(0), validators.lt(2)))
    random_seed: int = field(validator=instance_of(int))


@define(kw_only=True)
class SimulationConfigDC(SimulationConfig):
    """
    Configuration for simulating a Lévy-type process using the Dynamic Cutting (DC) approach.

    Attributes:
        h (float): Hyperparameter for dynamic cutting; must be > 0.
        eps (float): Hyperparameter for dynamic cutting; must be in (0, 1).
    """

    h: float = field(validator=validators.and_(instance_of(float), validators.gt(0)))
    eps: float = field(validator=validators.and_(instance_of(float), validators.gt(0), validators.lt(1)))


@define(kw_only=True)
class SimulationConfigAR(SimulationConfig):
    """
    Configuration for simulating a Lévy-type process using the Asmussen-Rosinski (AR) approach.

    Attributes:
        eps (float): Hyperparameter for accuracy; must be in (0, 1).
    """

    eps: float = field(validator=validators.and_(instance_of(float), validators.gt(0), validators.lt(1)))
