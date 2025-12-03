# run_training.py
"""
FedMO-DRLQ Training Script
==========================
Trains Rainbow DQN agent with error-aware state space and multi-objective rewards.
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

# ============================================================
# CONFIGURATION - Modify these for your experiments
# ============================================================

# Training parameters
NUM_EPISODES = 200
MAX_STEPS_PER_EPISODE = 100
LOG_INTERVAL = 10
SAVE_INTERVAL = 50

# Multi-objective weights (must sum to 1.0)
WEIGHT_TIME = 0.4       # Completion time importance
WEIGHT_FIDELITY = 0.4   # Quantum fidelity importance  
WEIGHT_ENERGY = 0.2     # Energy consumption importance

# Scalarization method: WEIGHTED_SUM, CHEBYSHEV, or CONSTRAINT_BASED
SCALARIZATION = ScalarizationMethod.WEIGHTED_SUM

# Output directory
OUTPUT_DIR = "outputs"

# ============================================================
# SETUP
# ============================================================

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join(OUTPUT_DIR, f"run_{timestamp}")
os.makedirs(run_dir, exist_ok=True)

print("=" * 60)
print("FedMO-DRLQ Training")
print("=" * 60)
print(f"Output directory: {run_dir}")
print()

# Configuration
config = FedMODRLQConfig()
config.multi_objective.weight_time = WEIGHT_TIME
config.multi_objective.weight_fidelity = WEIGHT_FIDELITY
config.multi_objective.weight_energy = WEIGHT_ENERGY
config.multi_objective.scalarization = SCALARIZATION

print("Configuration:")
print(f"  - Episodes: {NUM_EPISODES}")
print(f"  - Max steps/episode: {MAX_STEPS_PER_EPISODE}")
print(f"  - Weights: time={WEIGHT_TIME}, fidelity={WEIGHT_FIDELITY}, energy={WEIGHT_ENERGY}")
print(f"  - Scalarization: {SCALARIZATION.value}")
print()

# Save config
config_path = os.path.join(run_dir, "config.json")
with open(config_path, "w") as f:
    json.dump({
        "num_episodes": NUM_EPISODES,
        "max_steps": MAX_STEPS_PER_EPISODE,
        "weight_time": WEIGHT_TIME,
        "weight_fidelity": WEIGHT_FIDELITY,
        "weight_energy": WEIGHT_ENERGY,
        "scalarization": SCALARIZATION.value
    }, f, indent=2)

# Create environment
print("Creating environment...")
env = FedMODRLQEnv(config=config)
print(f"  - Observation space: {env.observation_space.shape}")
print(f"  - Action space: {env.action_space.n} QNodes")
print()

# Create agent
print("Creating Rainbow DQN agent...")
agent = RainbowDQNAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    device="cpu"  # Change to "cuda" if you have GPU
)
print(f"  - Device: cpu")
print()

# ============================================================
# TRAINING LOOP
# ============================================================

print("=" * 60)
print("Starting Training...")
print("=" * 60)
print()

# Metrics storage
all_rewards = []
all_fidelities = []
all_completion_times = []
all_energies = []

best_avg_reward = float('-inf')

for episode in range(NUM_EPISODES):
    # Reset environment
    obs, info = env.reset()
    episode_reward = 0
    episode_steps = 0
    
    # Episode loop
    for step in range(MAX_STEPS_PER_EPISODE):
        # Select action using agent
        action = agent.select_action(obs, training=True)
        
        # Execute action in environment
        next_obs, reward, done, truncated, info = env.step(action)
        
        # Store transition in replay buffer
        agent.store_transition(obs, action, reward, next_obs, done)
        
        # Update agent (train on batch from replay buffer)
        loss = agent.update()
        
        # Accumulate reward
        episode_reward += reward
        episode_steps += 1
        obs = next_obs
        
        if done or truncated:
            break
    
    # Get episode summary from environment
    summary = env.get_episode_summary()
    
    # Store metrics
    all_rewards.append(episode_reward)
    all_fidelities.append(summary.get('avg_fidelity', 0))
    all_completion_times.append(summary.get('total_completion_time', 0))
    all_energies.append(summary.get('total_energy', 0))
    
    # Logging
    if (episode + 1) % LOG_INTERVAL == 0:
        avg_reward = np.mean(all_rewards[-LOG_INTERVAL:])
        avg_fidelity = np.mean(all_fidelities[-LOG_INTERVAL:])
        avg_time = np.mean(all_completion_times[-LOG_INTERVAL:])
        
        print(f"Episode {episode+1:4d}/{NUM_EPISODES} | "
              f"Reward: {episode_reward:8.2f} | "
              f"Avg({LOG_INTERVAL}): {avg_reward:8.2f} | "
              f"Fidelity: {avg_fidelity:.4f} | "
              f"Time: {avg_time:.2f}")
        
        # Save best model
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            agent.save(os.path.join(run_dir, "best_agent.pt"))
    
    # Periodic save
    if (episode + 1) % SAVE_INTERVAL == 0:
        agent.save(os.path.join(run_dir, f"agent_ep{episode+1}.pt"))

# ============================================================
# SAVE FINAL RESULTS
# ============================================================

print()
print("=" * 60)
print("Training Complete!")
print("=" * 60)

# Save final agent
agent.save(os.path.join(run_dir, "final_agent.pt"))
print(f"✓ Final agent saved to {run_dir}/final_agent.pt")

# Save training metrics
metrics = {
    "episodes": list(range(1, NUM_EPISODES + 1)),
    "rewards": all_rewards,
    "fidelities": all_fidelities,
    "completion_times": all_completion_times,
    "energies": all_energies
}

metrics_path = os.path.join(run_dir, "training_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"✓ Metrics saved to {metrics_path}")

# Print summary
print()
print("Training Summary:")
print(f"  - Total episodes: {NUM_EPISODES}")
print(f"  - Best avg reward: {best_avg_reward:.2f}")
print(f"  - Final avg reward (last 10): {np.mean(all_rewards[-10:]):.2f}")
print(f"  - Final avg fidelity: {np.mean(all_fidelities[-10:]):.4f}")
print(f"  - Final avg completion time: {np.mean(all_completion_times[-10:]):.2f}")
print()
print(f"Results saved in: {run_dir}/")