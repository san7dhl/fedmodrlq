# pareto_analysis.py
"""
Pareto Front Analysis for FedMO-DRLQ
====================================
Generates Pareto-optimal solutions for time vs fidelity vs energy trade-offs.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from fedmo_drlq import (
    FedMODRLQConfig,
    FedMODRLQEnv,
    RainbowDQNAgent,
    ScalarizationMethod
)

OUTPUT_DIR = "outputs/pareto"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def is_pareto_optimal(costs):
    """Find Pareto-optimal points (minimization)."""
    is_optimal = np.ones(len(costs), dtype=bool)
    for i, c in enumerate(costs):
        if is_optimal[i]:
            # Check if any other point dominates this one
            is_optimal[is_optimal] = np.any(costs[is_optimal] < c, axis=1)
            is_optimal[i] = True
    return is_optimal

def run_with_weights(weight_time, weight_fidelity, weight_energy, num_episodes=50):
    """Run training with specific weights and return final metrics."""
    
    config = FedMODRLQConfig()
    config.multi_objective.weight_time = weight_time
    config.multi_objective.weight_fidelity = weight_fidelity
    config.multi_objective.weight_energy = weight_energy
    
    env = FedMODRLQEnv(config=config)
    agent = RainbowDQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device="cpu"
    )
    
    # Train
    for episode in range(num_episodes):
        obs, _ = env.reset()
        for step in range(100):
            action = agent.select_action(obs, training=True)
            next_obs, reward, done, _, _ = env.step(action)
            agent.store_transition(obs, action, reward, next_obs, done)
            agent.update()
            obs = next_obs
            if done:
                break
    
    # Evaluate
    eval_times = []
    eval_fidelities = []
    eval_energies = []
    
    for _ in range(10):  # 10 evaluation episodes
        obs, _ = env.reset()
        for step in range(100):
            action = agent.select_action(obs, training=False)
            obs, _, done, _, _ = env.step(action)
            if done:
                break
        summary = env.get_episode_summary()
        eval_times.append(summary.get('total_completion_time', 0))
        eval_fidelities.append(summary.get('avg_fidelity', 0))
        eval_energies.append(summary.get('total_energy', 0))
    
    return {
        "weights": (weight_time, weight_fidelity, weight_energy),
        "completion_time": np.mean(eval_times),
        "fidelity": np.mean(eval_fidelities),
        "energy": np.mean(eval_energies)
    }

# Generate weight combinations
print("=" * 60)
print("Pareto Front Analysis")
print("=" * 60)
print()

# Sample weights across the simplex
weight_combinations = [
    (1.0, 0.0, 0.0),  # Time only
    (0.0, 1.0, 0.0),  # Fidelity only
    (0.0, 0.0, 1.0),  # Energy only
    (0.5, 0.5, 0.0),  # Time + Fidelity
    (0.5, 0.0, 0.5),  # Time + Energy
    (0.0, 0.5, 0.5),  # Fidelity + Energy
    (0.33, 0.33, 0.34),  # Equal
    (0.6, 0.3, 0.1),
    (0.3, 0.6, 0.1),
    (0.1, 0.6, 0.3),
    (0.2, 0.4, 0.4),
    (0.4, 0.2, 0.4),
    (0.4, 0.4, 0.2),
    (0.7, 0.2, 0.1),
    (0.2, 0.7, 0.1),
]

results = []
for i, (wt, wf, we) in enumerate(weight_combinations):
    print(f"[{i+1}/{len(weight_combinations)}] Weights: time={wt:.2f}, fidelity={wf:.2f}, energy={we:.2f}")
    result = run_with_weights(wt, wf, we)
    results.append(result)
    print(f"  → Time: {result['completion_time']:.2f}, Fidelity: {result['fidelity']:.4f}, Energy: {result['energy']:.4f}")

# Find Pareto-optimal solutions
# For Pareto: minimize time, maximize fidelity (so minimize -fidelity), minimize energy
costs = np.array([
    [r['completion_time'], -r['fidelity'], r['energy']] 
    for r in results
])
pareto_mask = is_pareto_optimal(costs)

print(f"\n✓ Found {sum(pareto_mask)} Pareto-optimal solutions out of {len(results)}")

# Plot Pareto fronts
fig = plt.figure(figsize=(15, 5))

# Plot 1: Time vs Fidelity
ax1 = fig.add_subplot(131)
times = [r['completion_time'] for r in results]
fidelities = [r['fidelity'] for r in results]
ax1.scatter(times, fidelities, c='blue', alpha=0.5, label='All solutions')
ax1.scatter(
    [times[i] for i in range(len(times)) if pareto_mask[i]],
    [fidelities[i] for i in range(len(fidelities)) if pareto_mask[i]],
    c='red', s=100, marker='*', label='Pareto-optimal'
)
ax1.set_xlabel('Completion Time (lower is better)')
ax1.set_ylabel('Fidelity (higher is better)')
ax1.set_title('Time vs Fidelity Trade-off')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Time vs Energy
ax2 = fig.add_subplot(132)
energies = [r['energy'] for r in results]
ax2.scatter(times, energies, c='blue', alpha=0.5, label='All solutions')
ax2.scatter(
    [times[i] for i in range(len(times)) if pareto_mask[i]],
    [energies[i] for i in range(len(energies)) if pareto_mask[i]],
    c='red', s=100, marker='*', label='Pareto-optimal'
)
ax2.set_xlabel('Completion Time')
ax2.set_ylabel('Energy Consumption')
ax2.set_title('Time vs Energy Trade-off')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Fidelity vs Energy
ax3 = fig.add_subplot(133)
ax3.scatter(fidelities, energies, c='blue', alpha=0.5, label='All solutions')
ax3.scatter(
    [fidelities[i] for i in range(len(fidelities)) if pareto_mask[i]],
    [energies[i] for i in range(len(energies)) if pareto_mask[i]],
    c='red', s=100, marker='*', label='Pareto-optimal'
)
ax3.set_xlabel('Fidelity')
ax3.set_ylabel('Energy Consumption')
ax3.set_title('Fidelity vs Energy Trade-off')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pareto_fronts.png"), dpi=300)
plt.show()

# Save results
pareto_results = {
    "all_solutions": results,
    "pareto_optimal_indices": [i for i, p in enumerate(pareto_mask) if p],
    "pareto_solutions": [results[i] for i, p in enumerate(pareto_mask) if p]
}

with open(os.path.join(OUTPUT_DIR, "pareto_results.json"), "w") as f:
    json.dump(pareto_results, f, indent=2)

print(f"\n✓ Results saved to {OUTPUT_DIR}/")