# FedMO-DRLQ Folder Structure Guide

## Required Folder Structure

```
your_project/                          # Your project root
│
├── fedmo_drlq/                        # ← Place this entire folder
│   ├── __init__.py                    # Main package exports (230 lines)
│   ├── setup.py                       # Package installer
│   ├── requirements.txt               # Dependencies
│   ├── README.md                      # Documentation
│   ├── train.py                       # Training script (665 lines)
│   │
│   ├── core/                          # Core components (BASE LAYER)
│   │   ├── __init__.py                # Module exports (80 lines)
│   │   ├── config.py                  # Configuration classes (289 lines)
│   │   ├── error_metrics.py           # NISQ error handling (598 lines)
│   │   └── multi_objective_reward.py  # MORL rewards (618 lines)
│   │
│   ├── envs/                          # Gymnasium environments
│   │   ├── __init__.py                # Module exports (23 lines)
│   │   └── fedmo_env.py               # Main environment (774 lines)
│   │
│   ├── agents/                        # DRL agents
│   │   ├── __init__.py                # Module exports (39 lines)
│   │   ├── rainbow_dqn.py             # Rainbow DQN (759 lines)
│   │   └── ppo.py                     # PPO agent (444 lines)
│   │
│   ├── federated/                     # Federated learning
│   │   ├── __init__.py                # Module exports (38 lines)
│   │   └── federated_learning.py      # FL components (733 lines)
│   │
│   ├── evaluation/                    # Benchmarking
│   │   ├── __init__.py                # Module exports (35 lines)
│   │   ├── benchmark.py               # Benchmark runner (752 lines)
│   │   └── evaluator.py               # Evaluation utils (715 lines)
│   │
│   └── utils/                         # Utilities
│       ├── __init__.py                # Module exports (49 lines)
│       └── helpers.py                 # Helper functions (276 lines)
│
├── qsimpy/                            # Your existing QSimPy (optional)
│   └── ...
│
└── experiments/                       # Output directory
    └── ...
```

## Installation Options

### Option 1: Install as Package (Recommended)
```bash
cd your_project/fedmo_drlq
pip install -e .
```

### Option 2: Add to PYTHONPATH
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/your_project"
```

### Option 3: Direct Import (in your scripts)
```python
import sys
sys.path.insert(0, '/path/to/your_project')
from fedmo_drlq import FedMODRLQConfig
```

## Module Dependency Graph

```
                    ┌─────────────────┐
                    │   train.py      │  ← Entry point
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│    agents/      │ │   federated/    │ │  evaluation/    │
│  rainbow_dqn.py │ │ fed_learning.py │ │  benchmark.py   │
│     ppo.py      │ │                 │ │  evaluator.py   │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │     envs/       │
                    │  fedmo_env.py   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │     core/       │  ← Base layer (no deps)
                    │   config.py     │
                    │ error_metrics.py│
                    │  mo_reward.py   │
                    └─────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │     utils/      │
                    │   helpers.py    │
                    └─────────────────┘
```

## External Dependencies

| Package | Version | Used By |
|---------|---------|---------|
| `numpy` | >=1.21.0 | All modules |
| `torch` | >=1.12.0 | agents, federated, utils |
| `gymnasium` | >=0.28.0 | envs |
| `simpy` | >=4.0.0 | envs |

## Import Examples

### From Python Script
```python
# Full imports
from fedmo_drlq import (
    FedMODRLQConfig,
    FedMODRLQEnv,
    RainbowDQNAgent,
    PPOAgent,
    FederatedServer,
    BenchmarkRunner
)

# Selective imports
from fedmo_drlq.core import QNodeErrorMetrics, FidelityEstimator
from fedmo_drlq.agents import RainbowDQNAgent
from fedmo_drlq.federated import FedAvgAggregator
```

### From Command Line
```bash
# Run training
cd your_project
python -m fedmo_drlq.train --algorithm rainbow_dqn --episodes 100

# Or directly
python fedmo_drlq/train.py --help
```

## Integration with Existing QSimPy

If you have an existing QSimPy installation:

```
your_project/
├── qsimpy/                    # Your existing code
│   ├── resources/
│   │   ├── QNode.py
│   │   └── IBMQNode.py
│   ├── tasks/
│   │   └── QTask.py
│   └── brokers/
│       └── Broker.py
│
├── fedmo_drlq/               # New FedMO-DRLQ
│   └── ...
│
├── gymenv_qsimpy.py          # Your existing env
└── integration.py            # Bridge between them
```

### Integration Script Example
```python
# integration.py
from qsimpy.resources import QNode, IBMQNode
from qsimpy.tasks import QTask

from fedmo_drlq.core import (
    QNodeErrorMetrics,
    ErrorAwareStateSpace,
    MultiObjectiveRewardFunction
)

# Create error-aware wrapper for your existing QNodes
def enhance_qnode_with_errors(qnode: IBMQNode) -> QNodeErrorMetrics:
    return QNodeErrorMetrics.from_ibmq_error_dict(
        qnode_id=qnode.id,
        qnode_name=qnode.name,
        error_dict=qnode.error,
        system_dict=qnode.system_info
    )
```

## Verification

Run this to verify installation:

```python
# verify_install.py
import sys
try:
    from fedmo_drlq import __version__
    print(f"✓ FedMO-DRLQ {__version__} installed successfully")
    
    from fedmo_drlq import FedMODRLQConfig
    config = FedMODRLQConfig()
    print(f"✓ Config created: {config.experiment_name}")
    
    from fedmo_drlq import RainbowDQNAgent, PPOAgent
    print(f"✓ Agents available")
    
    from fedmo_drlq import FederatedServer
    print(f"✓ Federated components available")
    
    print("\n✅ All components working!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all dependencies are installed:")
    print("  pip install numpy torch gymnasium simpy")
```

## File Summary

| Category | Files | Total Lines |
|----------|-------|-------------|
| Core | 4 files | 1,585 |
| Agents | 3 files | 1,242 |
| Envs | 2 files | 797 |
| Federated | 2 files | 771 |
| Evaluation | 3 files | 1,502 |
| Utils | 2 files | 325 |
| Main | 2 files | 895 |
| **Total** | **18 files** | **7,117 lines** |
