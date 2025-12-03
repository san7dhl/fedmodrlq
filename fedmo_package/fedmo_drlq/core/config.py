"""
FedMO-DRLQ Configuration Module
================================
Central configuration for the FedMO-DRLQ framework including:
- NISQ device error parameters
- Multi-objective optimization weights
- Federated learning hyperparameters
- Environment settings

Author: Sandhya (NIT Sikkim)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np


class ObjectiveType(Enum):
    """Types of optimization objectives"""
    COMPLETION_TIME = "completion_time"
    FIDELITY = "fidelity"
    ENERGY = "energy"


class ScalarizationMethod(Enum):
    """Multi-objective scalarization strategies"""
    WEIGHTED_SUM = "weighted_sum"
    CHEBYSHEV = "chebyshev"
    CONSTRAINT_BASED = "constraint_based"


class FederatedAggregation(Enum):
    """Federated learning aggregation strategies"""
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    FEDNOVA = "fednova"


@dataclass
class NISQErrorConfig:
    """Configuration for NISQ device error characteristics"""
    
    # Two-qubit gate error bounds (typical range for superconducting qubits)
    two_qubit_error_min: float = 0.003  # 0.3%
    two_qubit_error_max: float = 0.015  # 1.5%
    
    # Single-qubit gate error bounds
    single_qubit_error_min: float = 0.0001  # 0.01%
    single_qubit_error_max: float = 0.001   # 0.1%
    
    # Coherence time bounds (microseconds)
    t1_min: float = 50.0    # 50 μs
    t1_max: float = 300.0   # 300 μs
    t2_min: float = 20.0    # 20 μs
    t2_max: float = 200.0   # 200 μs
    
    # Readout error bounds
    readout_error_min: float = 0.005  # 0.5%
    readout_error_max: float = 0.03   # 3%
    
    # Calibration drift timescale (milliseconds)
    calibration_drift_min: float = 10.0   # 10 ms
    calibration_drift_max: float = 100.0  # 100 ms
    
    # Error rate change bounds (per second)
    error_drift_rate_min: float = -0.001
    error_drift_rate_max: float = 0.001


@dataclass
class MultiObjectiveConfig:
    """Configuration for multi-objective optimization"""
    
    # Default weights for objectives (must sum to 1.0)
    weight_time: float = 0.4
    weight_fidelity: float = 0.4
    weight_energy: float = 0.2
    
    # Scalarization method
    scalarization: ScalarizationMethod = ScalarizationMethod.WEIGHTED_SUM
    
    # Constraint thresholds (for constraint-based scalarization)
    min_fidelity_threshold: float = 0.8  # Minimum acceptable fidelity
    max_energy_threshold: float = 1000.0  # Maximum energy (arbitrary units)
    
    # Chebyshev reference point (ideal values)
    ideal_time: float = 0.0
    ideal_fidelity: float = 1.0
    ideal_energy: float = 0.0
    
    # Reward scaling factors
    time_reward_scale: float = 1.0
    fidelity_reward_scale: float = 10.0  # Scale up fidelity importance
    energy_reward_scale: float = 0.1
    
    def validate_weights(self) -> bool:
        """Validate that weights sum to 1.0"""
        total = self.weight_time + self.weight_fidelity + self.weight_energy
        return np.isclose(total, 1.0, atol=1e-6)
    
    def get_weights(self) -> Tuple[float, float, float]:
        """Return weights as tuple"""
        return (self.weight_time, self.weight_fidelity, self.weight_energy)


@dataclass
class FederatedConfig:
    """Configuration for federated learning"""
    
    # Aggregation strategy
    aggregation: FederatedAggregation = FederatedAggregation.FEDAVG
    
    # Number of local training steps before aggregation
    local_steps: int = 10
    
    # Number of federated rounds
    num_rounds: int = 100
    
    # Client participation rate (fraction of clients per round)
    participation_rate: float = 1.0
    
    # FedProx proximal term coefficient
    fedprox_mu: float = 0.01
    
    # FedNova normalization
    fednova_tau_eff: float = 1.0
    
    # Communication compression (for bandwidth efficiency)
    gradient_compression: bool = False
    compression_ratio: float = 0.1
    
    # Differential privacy settings
    use_differential_privacy: bool = False
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    dp_clip_norm: float = 1.0
    
    # Asynchronous settings
    async_aggregation: bool = False
    staleness_threshold: int = 5


@dataclass
class EnvironmentConfig:
    """Configuration for the FedMO-DRLQ environment"""
    
    # Number of quantum nodes
    n_qnodes: int = 5
    
    # Number of tasks per episode
    n_qtasks: int = 25
    
    # Maximum simulation rounds
    max_rounds: int = 999
    
    # Observation space settings
    include_error_features: bool = True
    include_calibration_time: bool = True
    include_error_drift: bool = True
    
    # Observation normalization
    obs_normalization: str = "rescale_-1_1"
    
    # Reward settings
    reward_normalization: bool = True
    reward_clip: Optional[float] = 10.0
    
    # Task rescheduling penalty
    rescheduling_penalty: float = -0.1
    
    # Default shots for quantum circuits
    default_shots: int = 1024


@dataclass
class TrainingConfig:
    """Configuration for DRL training"""
    
    # Algorithm selection
    algorithm: str = "rainbow_dqn"  # Options: "rainbow_dqn", "ppo", "a2c"
    
    # Learning rate
    lr: float = 0.001
    
    # Discount factor
    gamma: float = 0.99
    
    # Batch size
    batch_size: int = 64
    
    # Replay buffer size
    buffer_size: int = 100000
    
    # Target network update frequency
    target_update_freq: int = 1000
    
    # Exploration settings (for DQN)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # PPO specific
    ppo_clip: float = 0.2
    ppo_epochs: int = 10
    
    # Rainbow DQN components
    use_double_dqn: bool = True
    use_dueling: bool = True
    use_prioritized_replay: bool = True
    use_noisy_nets: bool = True
    use_distributional: bool = True
    use_multistep: bool = True
    n_step: int = 3
    num_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    
    # Training iterations
    total_timesteps: int = 100000
    eval_frequency: int = 1000
    checkpoint_frequency: int = 10000


@dataclass
class FedMODRLQConfig:
    """Master configuration combining all sub-configurations"""
    
    nisq: NISQErrorConfig = field(default_factory=NISQErrorConfig)
    multi_objective: MultiObjectiveConfig = field(default_factory=MultiObjectiveConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Experiment metadata
    experiment_name: str = "fedmo_drlq_experiment"
    seed: int = 42
    device: str = "cpu"  # "cpu" or "cuda"
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.multi_objective.validate_weights():
            raise ValueError("Multi-objective weights must sum to 1.0")


# Pre-defined configurations for different scenarios
CONFIGS = {
    "default": FedMODRLQConfig(),
    
    "fidelity_focused": FedMODRLQConfig(
        multi_objective=MultiObjectiveConfig(
            weight_time=0.2,
            weight_fidelity=0.7,
            weight_energy=0.1
        )
    ),
    
    "time_focused": FedMODRLQConfig(
        multi_objective=MultiObjectiveConfig(
            weight_time=0.7,
            weight_fidelity=0.2,
            weight_energy=0.1
        )
    ),
    
    "energy_efficient": FedMODRLQConfig(
        multi_objective=MultiObjectiveConfig(
            weight_time=0.3,
            weight_fidelity=0.3,
            weight_energy=0.4
        )
    ),
    
    "federated_large_scale": FedMODRLQConfig(
        environment=EnvironmentConfig(n_qnodes=10, n_qtasks=50),
        federated=FederatedConfig(
            num_rounds=200,
            local_steps=20,
            use_differential_privacy=True
        )
    ),
}


def get_config(name: str = "default") -> FedMODRLQConfig:
    """Get a pre-defined configuration by name"""
    if name not in CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[name]
