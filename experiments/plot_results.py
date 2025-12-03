# plot_results.py
import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Get the run directory from command line or use latest
if len(sys.argv) > 1:
    run_dir = sys.argv[1]
else:
    # Find latest run
    output_dir = "outputs"
    runs = sorted([d for d in os.listdir(output_dir) if d.startswith("run_")])
    run_dir = os.path.join(output_dir, runs[-1])

print(f"Plotting results from: {run_dir}")

# Load metrics
with open(os.path.join(run_dir, "training_metrics.json"), "r") as f:
    metrics = json.load(f)

episodes = metrics["episodes"]
rewards = metrics["rewards"]
fidelities = metrics["fidelities"]
completion_times = metrics["completion_times"]

# Compute moving averages
window = 10
avg_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
avg_fidelities = np.convolve(fidelities, np.ones(window)/window, mode='valid')

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Rewards
axes[0, 0].plot(episodes, rewards, alpha=0.3, label='Raw')
axes[0, 0].plot(episodes[window-1:], avg_rewards, label=f'Avg({window})')
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Reward')
axes[0, 0].set_title('Training Rewards')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Fidelity
axes[0, 1].plot(episodes, fidelities, alpha=0.3, label='Raw')
axes[0, 1].plot(episodes[window-1:], avg_fidelities, label=f'Avg({window})')
axes[0, 1].set_xlabel('Episode')
axes[0, 1].set_ylabel('Fidelity')
axes[0, 1].set_title('Average Circuit Fidelity')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Completion Time
axes[1, 0].plot(episodes, completion_times)
axes[1, 0].set_xlabel('Episode')
axes[1, 0].set_ylabel('Time')
axes[1, 0].set_title('Total Completion Time')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Multi-objective (Fidelity vs Time)
axes[1, 1].scatter(completion_times, fidelities, c=episodes, cmap='viridis', alpha=0.6)
axes[1, 1].set_xlabel('Completion Time')
axes[1, 1].set_ylabel('Fidelity')
axes[1, 1].set_title('Fidelity vs Time Trade-off')
cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
cbar.set_label('Episode')

plt.tight_layout()
plt.savefig(os.path.join(run_dir, "training_plots.png"), dpi=150)
plt.show()

print(f"✓ Plot saved to {run_dir}/training_plots.png")