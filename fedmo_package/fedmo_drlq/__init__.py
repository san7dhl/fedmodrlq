"""
FedMO-DRLQ: Federated Multi-Objective Deep Reinforcement Learning 
for Quantum Cloud Computing

Author: Sandhya (NIT Sikkim)
"""

__version__ = "0.1.0"

# Core imports
from .core import (
    NISQDeviceMetrics,
    NISQErrorSimulator,
    FidelityEstimator,
    EnergyEstimator,
    IBM_DEVICE_CONFIGS,
    MultiObjectiveReward,
    MultiObjectiveConfig,
    ScalarizationMethod,
    FedMODRLQConfig,
    TrainingConfig,
    get_config,
    CONFIGS
)

# Agent imports
from .agents import (
    RainbowDQNAgent,
    RainbowConfig,
    PPOAgent
)

# Environment imports
from .envs import (
    FedMOEnv,
    make_fedmo_env
)

# Federated imports
from .federated import (
    FederatedAggregator,
    FederatedClient,
    FederatedTrainer,
    AggregationMethod
)

__all__ = [
    # Core
    'NISQDeviceMetrics',
    'NISQErrorSimulator',
    'FidelityEstimator',
    'EnergyEstimator',
    'IBM_DEVICE_CONFIGS',
    'MultiObjectiveReward',
    'MultiObjectiveConfig',
    'ScalarizationMethod',
    'FedMODRLQConfig',
    'TrainingConfig',
    'get_config',
    'CONFIGS',
    # Agents
    'RainbowDQNAgent',
    'RainbowConfig',
    'PPOAgent',
    # Environment
    'FedMOEnv',
    'make_fedmo_env',
    # Federated
    'FederatedAggregator',
    'FederatedClient',
    'FederatedTrainer',
    'AggregationMethod'
]