# ablation_study.py
"""
Ablation Study for FedMO-DRLQ
=============================
Tests contribution of each research gap component:
- Gap 1: Error-aware state space
- Gap 2: Multi-objective rewards
- Gap 3: Federated learning
"""

import os
import json
import numpy as np
from datetime import datetime

from fedmo_drlq import (
    FedMODRLQConfig,
    FedMODRLQEnv,
    RainbowDQNAgent,
    ScalarizationMethod
)

OUTPUT_DIR = "outputs/ablation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_EPISODES = 10000  # CRITICAL: Rainbow DQN needs 10k+ for convergence
NUM_SEEDS = 3  # Run each config with 3 random seeds

# Ablation configurations
ABLATION_CONFIGS = {
    "full_model": {
        "description": "Full FedMO-DRLQ (all components)",
        "error_aware": True,
        "multi_objective": True,
        "weights": (0.25, 0.6, 0.15)  # UPDATED: Fidelity-focused
    },
    "no_error_awareness": {
        "description": "Without error-aware state (Gap 1 disabled)",
        "error_aware": False,
        "multi_objective": True,
        "weights": (0.25, 0.6, 0.15)  # UPDATED: Fidelity-focused
    },
    "single_objective_time": {
        "description": "Single objective: time only (Gap 2 disabled)",
        "error_aware": True,
        "multi_objective": False,
        "weights": (1.0, 0.0, 0.0)
    },
    "single_objective_fidelity": {
        "description": "Single objective: fidelity only",
        "error_aware": True,
        "multi_objective": False,
        "weights": (0.0, 1.0, 0.0)
    },
    "equal_weights": {
        "description": "Equal weights (0.33, 0.33, 0.33)",
        "error_aware": True,
        "multi_objective": True,
        "weights": (0.33, 0.33, 0.34)
    },
    "fidelity_focused": {
        "description": "Fidelity-focused (0.2, 0.6, 0.2)",
        "error_aware": True,
        "multi_objective": True,
        "weights": (0.2, 0.6, 0.2)
    }
}

def run_experiment(config_name, config_params, seed):
    """Run single experiment with given configuration."""
    
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Create config
    config = FedMODRLQConfig()
    config.multi_objective.weight_time = config_params["weights"][0]
    config.multi_objective.weight_fidelity = config_params["weights"][1]
    config.multi_objective.weight_energy = config_params["weights"][2]
    config.environment.include_error_features = config_params["error_aware"]
    
    # Create environment and agent
    env = FedMODRLQEnv(config=config)
    agent = RainbowDQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device="cpu"
    )
    
    # Training
    rewards = []
    fidelities = []
    completion_times = []
    
    for episode in range(NUM_EPISODES):
        obs, _ = env.reset()
        episode_reward = 0
        
        for step in range(100):
            action = agent.select_action(obs, training=True)
            next_obs, reward, done, _, info = env.step(action)
            agent.store_transition(obs, action, reward, next_obs, done)
            agent.update()
            episode_reward += reward
            obs = next_obs
            if done:
                break
        
        summary = env.get_episode_summary()
        rewards.append(episode_reward)
        fidelities.append(summary.get('avg_fidelity', 0))
        completion_times.append(summary.get('total_completion_time', 0))
        
        # CRITICAL: Update learning rate and exploration decay each episode
        if hasattr(agent, 'update_learning_rate'):
            agent.update_learning_rate()
        if hasattr(agent, 'update_epsilon'):
            agent.update_epsilon()
    
    return {
        "final_reward": np.mean(rewards[-10:]),
        "final_fidelity": np.mean(fidelities[-10:]),
        "final_time": np.mean(completion_times[-10:]),
        "rewards": rewards,
        "fidelities": fidelities,
        "completion_times": completion_times
    }

# Run all ablation experiments
print("=" * 60)
print("FedMO-DRLQ Ablation Study")
print("=" * 60)
print()

all_results = {}

for config_name, config_params in ABLATION_CONFIGS.items():
    print(f"\nRunning: {config_name}")
    print(f"  {config_params['description']}")
    
    seed_results = []
    for seed in range(NUM_SEEDS):
        print(f"  Seed {seed + 1}/{NUM_SEEDS}...", end=" ")
        result = run_experiment(config_name, config_params, seed)
        seed_results.append(result)
        print(f"Reward: {result['final_reward']:.2f}, Fidelity: {result['final_fidelity']:.4f}")
    
    # Aggregate across seeds
    all_results[config_name] = {
        "description": config_params["description"],
        "reward_mean": np.mean([r["final_reward"] for r in seed_results]),
        "reward_std": np.std([r["final_reward"] for r in seed_results]),
        "fidelity_mean": np.mean([r["final_fidelity"] for r in seed_results]),
        "fidelity_std": np.std([r["final_fidelity"] for r in seed_results]),
        "time_mean": np.mean([r["final_time"] for r in seed_results]),
        "time_std": np.std([r["final_time"] for r in seed_results]),
    }

# Print summary table
print("\n" + "=" * 60)
print("ABLATION RESULTS SUMMARY")
print("=" * 60)
print(f"\n{'Configuration':<25} {'Reward':>15} {'Fidelity':>15} {'Time':>15}")
print("-" * 70)

for config_name, results in all_results.items():
    reward = f"{results['reward_mean']:.2f} ± {results['reward_std']:.2f}"
    fidelity = f"{results['fidelity_mean']:.4f} ± {results['fidelity_std']:.4f}"
    time = f"{results['time_mean']:.2f} ± {results['time_std']:.2f}"
    print(f"{config_name:<25} {reward:>15} {fidelity:>15} {time:>15}")

# Save results
results_path = os.path.join(OUTPUT_DIR, "ablation_results.json")
with open(results_path, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n✓ Results saved to {results_path}")