# FedMO-DRLQ: Federated Multi-Objective Deep Reinforcement Learning for Quantum Cloud Computing

## Overview

FedMO-DRLQ is a comprehensive framework addressing **four critical research gaps** in quantum cloud resource management:

| Gap | Problem | Solution |
|-----|---------|----------|
| **Gap 1** | No DRL approach incorporates NISQ error characteristics | Error-aware state space with T1/T2, gate errors, calibration drift |
| **Gap 2** | MORL never applied to quantum cloud scheduling | Multi-objective rewards with weighted sum, Chebyshev, constraint-based |
| **Gap 3** | No federated learning for quantum cloud | FedAvg, FedProx, FedNova aggregation strategies |
| **Gap 4** | Limited scalability analysis | Scalable federated architecture with QSimPy integration |

## Installation

### Requirements
```bash
pip install torch numpy gymnasium simpy
```

### Setup
Place the `fedmo_drlq` folder in your project directory or install as a package:
```bash
# Option 1: Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/fedmo_drlq"

# Option 2: Install in development mode
pip install -e /path/to/fedmo_drlq
```

## Project Structure

```
fedmo_drlq/
├── __init__.py                     # Main package exports
├── train.py                        # Training script with CLI
│
├── core/                           # Core components
│   ├── config.py                   # Configuration dataclasses
│   ├── error_metrics.py            # NISQ error handling (Gap 1)
│   └── multi_objective_reward.py   # MORL rewards (Gap 2)
│
├── envs/                           # Gymnasium environments
│   └── fedmo_env.py                # FedMO-DRLQ environment
│
├── agents/                         # DRL agents
│   ├── rainbow_dqn.py              # Rainbow DQN (6 extensions)
│   └── ppo.py                      # Proximal Policy Optimization
│
├── federated/                      # Federated learning (Gap 3)
│   └── federated_learning.py       # Server, Client, Aggregators
│
├── evaluation/                     # Benchmarking (Gap 4)
│   ├── benchmark.py                # Benchmark runner
│   └── evaluator.py                # Evaluation utilities
│
└── utils/                          # Utility functions
    └── helpers.py                  # Logging, metrics, schedulers
```

## Files Created in This Session

| File | Description | Gap Addressed |
|------|-------------|---------------|
| `core/config.py` | Configuration classes for all components | All |
| `core/error_metrics.py` | NISQ error handling, fidelity estimation | Gap 1 |
| `core/multi_objective_reward.py` | MORL reward functions, scalarization | Gap 2 |
| `envs/fedmo_env.py` | Enhanced Gymnasium environment | Gap 1, 2 |
| `agents/rainbow_dqn.py` | Rainbow DQN with all 6 extensions | - |
| `agents/ppo.py` | PPO agent implementation | - |
| `federated/federated_learning.py` | FedAvg, FedProx, FedNova | Gap 3 |
| `evaluation/benchmark.py` | Benchmarking against baselines | Gap 4 |

## Quick Start

### Basic Usage
```python
from fedmo_drlq import (
    FedMODRLQConfig,
    FedMODRLQEnv,
    RainbowDQNAgent,
    MultiObjectiveRewardFunction
)

# Create configuration
config = FedMODRLQConfig()
config.multi_objective.weight_time = 0.4
config.multi_objective.weight_fidelity = 0.4
config.multi_objective.weight_energy = 0.2

# Create environment
env = FedMODRLQEnv(config=config)

# Create agent
agent = RainbowDQNAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    config=config.training
)

# Training loop
obs, info = env.reset()
for step in range(10000):
    action = agent.select_action(obs)
    next_obs, reward, done, truncated, info = env.step(action)
    agent.store_transition(obs, action, reward, next_obs, done)
    agent.update()
    obs = next_obs
    if done:
        obs, info = env.reset()
```

### Command Line Training
```bash
# Single agent training with Rainbow DQN
python train.py --algorithm rainbow_dqn --episodes 1000

# PPO training with fidelity focus
python train.py --algorithm ppo --weight-fidelity 0.7 --weight-time 0.2

# Federated training
python train.py --federated --num-clients 5 --fed-rounds 100

# Benchmark only
python train.py --benchmark-only --eval-episodes 20
```

## Key Components

### 1. Error-Aware State Space (Gap 1)
```python
from fedmo_drlq.core import QNodeErrorMetrics, FidelityEstimator

# Create error metrics for IBM backends
metrics = QNodeErrorMetrics.from_ibmq_error_dict(
    qnode_id=0,
    qnode_name="washington",
    error_dict={"T1": 91.36, "T2": 93.05, "CNOT_error": "126_112:0.0179"},
    system_dict={"qubits": 127, "qv": 64, "clops": 850}
)

# Get normalized feature vector (7 features)
features = metrics.get_feature_vector(normalize=True)
# [two_qubit_error, t2, readout_error, calibration_staleness, 
#  error_drift_rate, quantum_volume, clops]

# Estimate circuit fidelity
estimator = FidelityEstimator()
fidelity = estimator.estimate_circuit_fidelity(
    metrics, circuit_depth=50, n_qubits=10
)
```

### 2. Multi-Objective Reward (Gap 2)
```python
from fedmo_drlq.core import (
    MultiObjectiveRewardFunction,
    ScalarizationMethod,
    MultiObjectiveConfig
)

# Configure objectives
mo_config = MultiObjectiveConfig(
    weight_time=0.4,
    weight_fidelity=0.4,
    weight_energy=0.2,
    scalarization=ScalarizationMethod.CHEBYSHEV
)

reward_fn = MultiObjectiveRewardFunction(mo_config)

# Compute reward
scalar_reward, components = reward_fn.compute_reward(
    waiting_time=5.0,
    execution_time=10.0,
    error_metrics=metrics,
    circuit_depth=50,
    n_qubits=10
)

print(f"Time reward: {components.time_reward}")
print(f"Fidelity reward: {components.fidelity_reward}")
print(f"Energy reward: {components.energy_reward}")
print(f"Total: {scalar_reward}")
```

### 3. Federated Learning (Gap 3)
```python
from fedmo_drlq.federated import (
    FederatedServer,
    FederatedClient,
    FedProxAggregator
)

# Create server with FedProx aggregation
config.federated.aggregation = FederatedAggregation.FEDPROX
config.federated.fedprox_mu = 0.01

server = FederatedServer(global_model, config.federated)

# Create clients (one per quantum datacenter)
clients = [
    FederatedClient(client_id=i, local_model=model, optimizer=opt)
    for i in range(5)
]

# Federated round
for round in range(100):
    global_params = server.get_global_parameters()
    
    for client in clients:
        client.set_global_parameters(global_params)
        # ... local training ...
        update = client.get_update(round)
        server.receive_update(update)
    
    server.aggregate_round()
```

### 4. Benchmarking (Gap 4)
```python
from fedmo_drlq.evaluation import (
    BenchmarkRunner,
    FCFSScheduler,
    SJFScheduler,
    GreedyErrorScheduler
)

# Create benchmark runner
runner = BenchmarkRunner(env_factory=lambda: FedMODRLQEnv())

# Run all baselines
results = runner.run_all_baselines(num_episodes=10)

# Compare results
runner.print_summary()

# Benchmark your trained agent
agent_results = runner.benchmark_agent(
    agent=trained_agent,
    agent_name="FedMO-DRLQ",
    num_episodes=10
)
```

## Configuration Options

### Multi-Objective Scalarization
| Method | Description | Use Case |
|--------|-------------|----------|
| `WEIGHTED_SUM` | Linear combination | Simple, convex Pareto fronts |
| `CHEBYSHEV` | Min-max weighted deviation | Non-convex Pareto regions |
| `CONSTRAINT_BASED` | Primary + constraints | Hard fidelity/energy limits |

### Federated Aggregation
| Strategy | Description | Best For |
|----------|-------------|----------|
| `FEDAVG` | Simple averaging | IID data |
| `FEDPROX` | Proximal regularization | Non-IID data |
| `FEDNOVA` | Normalized averaging | Varying local steps |

### Rainbow DQN Components
All 6 extensions enabled by default:
- Double DQN (reduces overestimation)
- Prioritized Experience Replay
- Dueling Networks
- Multi-step Learning (n=3)
- Distributional RL (51 atoms)
- Noisy Nets (exploration)

## Integration with QSimPy

To integrate with your existing QSimPy codebase:

```python
# In your existing gymenv_qsimpy.py, extend the observation space:
from fedmo_drlq.core import ErrorAwareStateSpace

state_space = ErrorAwareStateSpace(
    n_qnodes=5,
    include_error_features=True
)

# Use the enhanced observation
obs = state_space.construct_observation(
    qtask_obs=current_task_features,
    qnodes=self.qnodes,
    current_qtask=self.current_qtask
)
```

## Research Paper Integration

This implementation directly addresses the gaps identified in your research gap analysis document:

1. **Section 2 (Gap 1)**: `error_metrics.py` implements the error-aware state space formula from Equation (1)
2. **Section 3 (Gap 2)**: `multi_objective_reward.py` implements the reward function from Equation (2)
3. **Section 4 (Gap 3)**: `federated_learning.py` implements the federated architecture
4. **Section 5 (Gap 4)**: `benchmark.py` provides scalable evaluation

## Citation

If you use this code in your research, please cite:
```bibtex
@article{sandhya2025fedmodrlq,
  title={FedMO-DRLQ: Federated Multi-Objective Deep Reinforcement Learning 
         for Quantum Cloud Computing Resource Management},
  author={Sandhya},
  journal={[Target Q1 Journal]},
  year={2025}
}
```

## Author

**Sandhya**  
PhD Student, NIT Sikkim  
Research: Quantum Cloud Computing, Deep Reinforcement Learning, Federated Learning
