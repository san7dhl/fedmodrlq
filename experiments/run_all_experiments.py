# experiments/run_all_experiments.py
# FedMO-DRLQ Comprehensive Experiments
# Refactored to use package environment and loop through all datasets

import os
import sys
import glob  # Added for dataset looping
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import warnings
import torch

warnings.filterwarnings('ignore')

# Auto-detect device (use GPU if available)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Device] Using: {DEVICE}")

# Add fedmo_package to path
# Assuming this script is in experiments/ and fedmo_package is in the root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fedmo_package'))

# Imports from fedmo_drlq package
from fedmo_drlq.core.error_metrics import (
    NISQDeviceMetrics, NISQErrorSimulator, FidelityEstimator,
    EnergyEstimator, IBM_DEVICE_CONFIGS
)
from fedmo_drlq.core.multi_objective_reward import (
    MultiObjectiveReward, MultiObjectiveConfig, ScalarizationMethod
)
from fedmo_drlq.agents.rainbow_dqn import RainbowDQNAgent, RainbowConfig
from fedmo_drlq.agents.ppo import PPOAgent, TrainingConfig as PPOConfig  # Added for PPO baseline
from fedmo_drlq.federated.federated_learning import (
    FederatedAggregator, FederatedTrainer, FederatedConfig, AggregationMethod
)
# CRITICAL FIX: Import the real environment instead of defining SimplifiedEnv
from fedmo_drlq.envs.fedmo_env import FedMOEnv

@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    # Dataset
    dataset_path: str = ""
    
    # Environment
    n_qnodes: int = 5
    n_qtasks_per_episode: int = 25
    
    # Training - INCREASED for proper Rainbow DQN convergence  
    n_episodes: int = 50000  # Was 10000 - need 50k for publication-quality results
    n_eval_episodes: int = 100
    
    # For quick testing (set via command line)
    quick_mode: bool = False
    n_episodes_quick: int = 5000
    
    # Federated - INCREASED for convergence
    n_datacenters: int = 3
    federated_rounds: int = 50   # Was 20 - need more for convergence
    local_steps: int = 20        # Was 10 - more local training per round
    
    # Output
    output_dir: str = "outputs"
    seed: int = 42

class HeuristicScheduler:
    """Heuristic scheduling algorithms for baseline comparison."""
    
    def __init__(self, n_qnodes: int = 5):
        self.n_qnodes = n_qnodes
        # IBM node capacities: washington(127), kolkata(27), hanoi(27), perth(7), lagos(7)
        self.node_qubits = [127, 27, 27, 7, 7][:n_qnodes]
        self.rr_index = 0
    
    def reset(self):
        self.rr_index = 0
    
    def _get_task_qubits(self, state: np.ndarray) -> int:
        """Extract task qubit requirement from state (denormalize)."""
        # state[1] is qubit_number / 127.0
        return max(1, int(state[1] * 127))
    
    def _get_valid_nodes(self, state: np.ndarray) -> list:
        """Get nodes that can handle current task."""
        task_qubits = self._get_task_qubits(state)
        valid = [i for i, q in enumerate(self.node_qubits) if q >= task_qubits]
        # If no valid nodes, return node with most qubits (best effort)
        if not valid:
            return [0]  # washington has 127 qubits
        return valid
    
    def _get_node_times(self, state: np.ndarray) -> list:
        """Extract next available times for all nodes."""
        per_node_dim = (len(state) - 4) // self.n_qnodes
        times = []
        for i in range(self.n_qnodes):
            start_idx = 4 + i * per_node_dim
            times.append(state[start_idx + per_node_dim - 1])
        return times
    
    def _get_node_errors(self, state: np.ndarray) -> list:
        """Extract error rates for all nodes."""
        per_node_dim = (len(state) - 4) // self.n_qnodes
        errors = []
        for i in range(self.n_qnodes):
            start_idx = 4 + i * per_node_dim
            errors.append(state[start_idx])
        return errors
    
    def random(self, state: np.ndarray) -> int:
        """Random scheduling among valid nodes."""
        valid = self._get_valid_nodes(state)
        return np.random.choice(valid)
    
    def round_robin(self, state: np.ndarray) -> int:
        """Round-robin among valid nodes."""
        valid = self._get_valid_nodes(state)
        # Find next valid node
        for _ in range(self.n_qnodes):
            if self.rr_index in valid:
                action = self.rr_index
                self.rr_index = (self.rr_index + 1) % self.n_qnodes
                return action
            self.rr_index = (self.rr_index + 1) % self.n_qnodes
        return valid[0]
    
    def min_completion_time(self, state: np.ndarray) -> int:
        """MCT: Pick valid node with minimum next available time."""
        valid = self._get_valid_nodes(state)
        times = self._get_node_times(state)
        
        best_node = valid[0]
        best_time = float('inf')
        for i in valid:
            if times[i] < best_time:
                best_time = times[i]
                best_node = i
        return best_node
    
    def min_error(self, state: np.ndarray) -> int:
        """Pick valid node with minimum error rate."""
        valid = self._get_valid_nodes(state)
        errors = self._get_node_errors(state)
        
        best_node = valid[0]
        best_error = float('inf')
        for i in valid:
            if errors[i] < best_error:
                best_error = errors[i]
                best_node = i
        return best_node
    
    def best_fidelity(self, state: np.ndarray) -> int:
        """Pick valid node with best fidelity score (high T2, low error)."""
        valid = self._get_valid_nodes(state)
        per_node_dim = (len(state) - 4) // self.n_qnodes
        
        best_node = valid[0]
        best_score = -float('inf')
        for i in valid:
            start_idx = 4 + i * per_node_dim
            error = state[start_idx]
            t2 = state[start_idx + 1]
            score = t2 / (error + 1e-6)
            if score > best_score:
                best_score = score
                best_node = i
        return best_node

def run_heuristic_evaluation(
    env: FedMOEnv,
    scheduler: HeuristicScheduler,
    policy_name: str,
    policy_fn,
    n_episodes: int,
    max_steps_per_episode: int = 200  # ADD THIS PARAMETER
) -> pd.DataFrame:
    """Run evaluation for a heuristic policy with step limit to prevent infinite loops."""
    results = []
    
    for ep in range(n_episodes):
        scheduler.reset()
        obs, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < max_steps_per_episode:
            action = policy_fn(obs)
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            if truncated:
                done = True
        
        summary = env.get_episode_summary()
        summary['episode'] = ep
        summary['policy'] = policy_name
        summary['steps'] = steps
        summary['completed'] = done
        results.append(summary)
        
    return pd.DataFrame(results)

def run_drl_training(
    env: FedMOEnv,
    agent: RainbowDQNAgent,
    n_episodes: int,
    eval_interval: int = 10,
    verbose: bool = True,
    max_steps_per_episode: int = 200  # ADD: Step limit to prevent infinite loops
) -> Tuple[List[Dict], RainbowDQNAgent]:
    """Train DRL agent with step limit to prevent infinite loops."""
    training_history = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        ep_steps = 0
        
        while not done and ep_steps < max_steps_per_episode:
            # Get action mask to prevent invalid actions
            action_mask = env.get_action_mask() if hasattr(env, 'get_action_mask') else None
            action = agent.select_action(obs, training=True, action_mask=action_mask)
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Merge done and truncated for training signal
            training_done = done or truncated
            
            agent.store_transition(obs, action, reward, next_obs, training_done)
            
            # GPU OPTIMIZATION: Multiple gradient updates per step
            if len(agent.replay_buffer) >= agent.config.min_buffer_size:
                for _ in range(16):  # 16 updates per step - higher GPU utilization
                    agent.update()
            
            obs = next_obs
            ep_reward += reward
            ep_steps += 1
            
            if truncated: 
                done = True
                
        summary = env.get_episode_summary()
        # Provide defaults if summary is empty (no tasks completed before step limit)
        if not summary:
            summary = {
                'total_reward': ep_reward,
                'mean_reward': ep_reward / max(ep_steps, 1),
                'mean_fidelity': 0.0,
                'mean_completion_time': 0.0,
                'total_energy': 0.0,
                'n_tasks': 0,
                'pareto_front_size': 0
            }
        summary['episode'] = ep
        summary['steps'] = ep_steps
        summary['training_loss'] = agent.losses[-1] if hasattr(agent, 'losses') and agent.losses else 0
        training_history.append(summary)
        
        # CRITICAL: Update learning rate and exploration decay each episode
        if hasattr(agent, 'update_learning_rate'):
            agent.update_learning_rate()  # Decay LR from 0.0001 → 0.00001
        if hasattr(agent, 'update_epsilon'):
            agent.update_epsilon()  # Decay exploration from 0.3 → 0.01
        
        if verbose and (ep + 1) % eval_interval == 0:
            print(f"Episode {ep+1}/{n_episodes}: "
                  f"Reward={ep_reward:.2f}, "
                  f"Fidelity={summary.get('mean_fidelity', 0):.3f}, "
                  f"Time={summary.get('mean_completion_time', 0):.2f}")
            
    return training_history, agent

def run_experiments(config: ExperimentConfig):
    """Run all experiments for a specific dataset"""
    
    # Create output directory
    dataset_name = os.path.basename(config.dataset_path).replace('.csv', '')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.output_dir, f"run_{dataset_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"FedMO-DRLQ Experiments - {dataset_name}")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Dataset: {config.dataset_path}")
    print(f"Seed: {config.seed}")
    print()
    
    # Set seeds
    np.random.seed(config.seed)
    
    # ========== EXPERIMENT 1: Baseline Heuristics ==========
    print("=" * 60)
    print("EXPERIMENT 1: Baseline Heuristics Evaluation")
    print("=" * 60)
    
    # Use real FedMOEnv
    env = FedMOEnv(
        dataset_path=config.dataset_path,
        n_qnodes=config.n_qnodes,
        n_qtasks_per_episode=config.n_qtasks_per_episode,
        seed=config.seed,
        include_error_features=True # Heuristics can use error features
    )
    
    scheduler = HeuristicScheduler(n_qnodes=config.n_qnodes)
    
    heuristic_policies = {
        'Random': scheduler.random,
        'RoundRobin': scheduler.round_robin,
        'MinCompletionTime': scheduler.min_completion_time,
        'MinError': scheduler.min_error,
        'BestFidelity': scheduler.best_fidelity
    }
    
    all_heuristic_results = []
    
    for name, policy_fn in heuristic_policies.items():
        print(f"\nEvaluating {name}...")
        results = run_heuristic_evaluation(
            env, scheduler, name, policy_fn, config.n_eval_episodes
        )
        all_heuristic_results.append(results)
        
        mean_reward = results['mean_reward'].mean()
        mean_fidelity = results['mean_fidelity'].mean()
        mean_time = results['mean_completion_time'].mean()
        
        print(f"  Mean Reward: {mean_reward:.4f}")
        print(f"  Mean Fidelity: {mean_fidelity:.4f}")
        print(f"  Mean Completion Time: {mean_time:.2f}s")
        
    heuristics_df = pd.concat(all_heuristic_results)
    heuristics_df.to_csv(os.path.join(output_dir, "heuristics_results.csv"), index=False)
    
    # ========== EXPERIMENT 2: DRL Training ==========
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Rainbow DQN Training")
    print("=" * 60)
    
    # Standard DRL Env
    env_drl = FedMOEnv(
        dataset_path=config.dataset_path,
        n_qnodes=config.n_qnodes,
        n_qtasks_per_episode=config.n_qtasks_per_episode,
        include_error_features=True,
        seed=config.seed
    )
    
    rainbow_config = RainbowConfig(
        # FAST CONVERGENCE CONFIG - 3-5x faster training
        hidden_dim=128,           # FAST: Smaller network
        lr=0.0003,                # FAST: Higher LR
        lr_decay=0.9995,          # FAST: Faster decay
        lr_min=0.00005,           # FAST: Higher minimum
        gamma=0.99,
        batch_size=256,           # GPU: Larger batches
        buffer_size=100000,       # GPU: Large buffer
        min_buffer_size=2000,     # FAST: Start learning earlier
        target_update_freq=500,   # FAST: More frequent (for hard update fallback)
        use_soft_update=True,     # FAST: Polyak averaging
        tau=0.005,                # FAST: Soft update rate
        use_double_dqn=True,      # KEEP: Essential for stability
        use_prioritized_replay=True,   # KEEP: Helps sparse rewards
        use_dueling=True,         # KEEP: Better value estimation
        use_multistep=True,
        n_step=3,                 # FAST: Shorter horizon
        use_distributional=False, # FAST: Disabled for speed
        use_noisy_nets=False,     # FAST: Use epsilon-greedy instead
        use_hybrid_exploration=False,  # FAST: Disabled
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=30000       # FAST: Faster exploration decay
    )
    
    agent = RainbowDQNAgent(
        state_dim=env_drl.observation_space.shape[0],
        action_dim=config.n_qnodes,
        config=rainbow_config,
        device=DEVICE
    )
    
    # GPU OPTIMIZATION: Compile model for faster execution (PyTorch 2.0+)
    if hasattr(torch, 'compile') and DEVICE == 'cuda':
        try:
            agent.online_net = torch.compile(agent.online_net, mode='reduce-overhead')
            agent.target_net = torch.compile(agent.target_net, mode='reduce-overhead')
            print("[GPU] Models compiled with torch.compile")
        except Exception as e:
            print(f"[GPU] Compilation skipped: {e}")
    
    print("\nTraining Rainbow DQN with error-aware features...")
    drl_history, trained_agent = run_drl_training(
        env_drl, agent, config.n_episodes, 
        eval_interval=500, verbose=True  # GPU OPTIMIZED: Reduced logging
    )
    
    drl_df = pd.DataFrame(drl_history)
    drl_df['policy'] = 'RainbowDQN_ErrorAware'
    drl_df.to_csv(os.path.join(output_dir, "rainbow_dqn_results.csv"), index=False)
    
    # Save trained agent
    trained_agent.save(os.path.join(output_dir, "rainbow_dqn_agent.pt"))
    
    # ========== EXPERIMENT 2b: PPO Baseline ==========
    print("\n" + "=" * 60)
    print("EXPERIMENT 2b: PPO Baseline Training")
    print("=" * 60)
    
    # Create PPO config
    ppo_config = PPOConfig(
        algorithm="ppo",
        lr=0.0003,
        gamma=0.99,
        batch_size=64,
        ppo_clip=0.2,
        ppo_epochs=10,
    )
    
    env_ppo = FedMOEnv(
        dataset_path=config.dataset_path,
        n_qnodes=config.n_qnodes,
        n_qtasks_per_episode=config.n_qtasks_per_episode,
        include_error_features=True,
        seed=config.seed + 500
    )
    
    ppo_agent = PPOAgent(
        state_dim=env_ppo.observation_space.shape[0],
        action_dim=config.n_qnodes,
        config=ppo_config,
        device=DEVICE
    )
    
    print("\nTraining PPO baseline...")
    
    ppo_history = []
    for ep in range(config.n_episodes):
        obs, _ = env_ppo.reset()
        done = False
        truncated = False
        ep_reward = 0
        ep_steps = 0
        max_steps = 200
        
        while not done and not truncated and ep_steps < max_steps:
            action, log_prob, value = ppo_agent.select_action(obs, training=True)
            
            # Action masking
            if hasattr(env_ppo, 'get_action_mask'):
                mask = env_ppo.get_action_mask()
                if mask is not None and not mask[action]:
                    valid_actions = np.where(mask)[0]
                    if len(valid_actions) > 0:
                        action = np.random.choice(valid_actions)
            
            next_obs, reward, done, truncated, info = env_ppo.step(action)
            ppo_agent.store_transition(obs, action, reward, value, log_prob, done or truncated)
            obs = next_obs
            ep_reward += reward
            ep_steps += 1
        
        # Update PPO when buffer has enough samples
        if len(ppo_agent.buffer) >= ppo_config.batch_size:
            ppo_agent.update()
        
        summary = env_ppo.get_episode_summary()
        summary['episode'] = ep
        summary['episode_reward'] = ep_reward
        ppo_history.append(summary)
        
        if (ep + 1) % 1000 == 0:
            recent = [h.get('mean_reward', 0) for h in ppo_history[-100:]]
            print(f"  Episode {ep+1}/{config.n_episodes}: Reward={np.mean(recent):.3f}")
    
    ppo_df = pd.DataFrame(ppo_history)
    ppo_df['policy'] = 'PPO'
    ppo_df.to_csv(os.path.join(output_dir, "ppo_results.csv"), index=False)
    
    ppo_final_fidelity = np.mean([h.get('mean_fidelity', 0) for h in ppo_history[-100:]])
    ppo_final_time = np.mean([h.get('mean_completion_time', 0) for h in ppo_history[-100:]])
    print(f"\nPPO Final: Fidelity={ppo_final_fidelity:.4f}, Time={ppo_final_time:.1f}s")
    
    # ========== EXPERIMENT 2c: Standard DQN Baseline ==========
    print("\n" + "=" * 60)
    print("EXPERIMENT 2c: Standard DQN Baseline Training")
    print("=" * 60)
    
    # Standard DQN: Rainbow with all extensions DISABLED (for fair comparison)
    standard_dqn_config = RainbowConfig(
        hidden_dim=128,           # FAST: Match Rainbow config
        lr=0.001,
        gamma=0.99,
        batch_size=256,           # GPU: Larger batches
        buffer_size=100000,       # GPU: Large buffer
        min_buffer_size=2000,     # FAST: Start learning earlier
        target_update_freq=500,
        use_soft_update=True,     # FAST: Soft updates
        tau=0.005,
        use_double_dqn=True,      # Keep only Double DQN
        use_prioritized_replay=False,
        use_dueling=False,
        use_multistep=False,
        use_distributional=False,
        use_noisy_nets=False,
        use_hybrid_exploration=False,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=30000       # FAST: Was 100000
    )
    
    env_dqn = FedMOEnv(
        dataset_path=config.dataset_path,
        n_qnodes=config.n_qnodes,
        n_qtasks_per_episode=config.n_qtasks_per_episode,
        include_error_features=True,
        seed=config.seed + 1000
    )
    
    dqn_agent = RainbowDQNAgent(
        state_dim=env_dqn.observation_space.shape[0],
        action_dim=config.n_qnodes,
        config=standard_dqn_config,
        device=DEVICE
    )
    
    print("\nTraining Standard DQN baseline...")
    dqn_history, _ = run_drl_training(
        env_dqn, dqn_agent, config.n_episodes,
        eval_interval=500, verbose=True  # FIXED: Enable progress output
    )
    
    dqn_df = pd.DataFrame(dqn_history)
    dqn_df['policy'] = 'StandardDQN'
    dqn_df.to_csv(os.path.join(output_dir, "standard_dqn_results.csv"), index=False)
    
    dqn_final_fidelity = np.mean([h.get('mean_fidelity', 0) for h in dqn_history[-100:]])
    dqn_final_time = np.mean([h.get('mean_completion_time', 0) for h in dqn_history[-100:]])
    print(f"\nStandard DQN Final: Fidelity={dqn_final_fidelity:.4f}, Time={dqn_final_time:.1f}s")
    
    # ========== DRL COMPARISON SUMMARY ==========
    print("\n" + "=" * 60)
    print("DRL ALGORITHMS COMPARISON")
    print("=" * 60)
    
    rainbow_final_fidelity = np.mean([h.get('mean_fidelity', 0) for h in drl_history[-100:]])
    rainbow_final_time = np.mean([h.get('mean_completion_time', 0) for h in drl_history[-100:]])
    rainbow_final_reward = np.mean([h.get('mean_reward', 0) for h in drl_history[-100:]])
    ppo_final_reward = np.mean([h.get('mean_reward', 0) for h in ppo_history[-100:]])
    dqn_final_reward = np.mean([h.get('mean_reward', 0) for h in dqn_history[-100:]])
    
    print(f"\n{'Algorithm':<20} {'Fidelity':>10} {'Time (s)':>12} {'Reward':>10}")
    print("-" * 55)
    print(f"{'Rainbow DQN (Ours)':<20} {rainbow_final_fidelity:>10.4f} {rainbow_final_time:>12.1f} {rainbow_final_reward:>10.4f}")
    print(f"{'PPO':<20} {ppo_final_fidelity:>10.4f} {ppo_final_time:>12.1f} {ppo_final_reward:>10.4f}")
    print(f"{'Standard DQN':<20} {dqn_final_fidelity:>10.4f} {dqn_final_time:>12.1f} {dqn_final_reward:>10.4f}")
    
    # Save comparison
    comparison_df = pd.DataFrame([
        {'Algorithm': 'RainbowDQN', 'Fidelity': rainbow_final_fidelity, 'Time': rainbow_final_time, 'Reward': rainbow_final_reward},
        {'Algorithm': 'PPO', 'Fidelity': ppo_final_fidelity, 'Time': ppo_final_time, 'Reward': ppo_final_reward},
        {'Algorithm': 'StandardDQN', 'Fidelity': dqn_final_fidelity, 'Time': dqn_final_time, 'Reward': dqn_final_reward}
    ])
    comparison_df.to_csv(os.path.join(output_dir, "drl_comparison.csv"), index=False)
    
    # ========== EXPERIMENT 3: Multi-Objective Comparison ==========
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Multi-Objective Scalarization Comparison")
    print("=" * 60)
    
    # FIXED: Train SEPARATE agents for each objective configuration
    # This properly demonstrates the multi-objective tradeoffs
    
    mo_training_configs = [
        {
            'name': 'TimeFocused',
            'weights': (0.60, 0.25, 0.15),
            'scalarization': ScalarizationMethod.WEIGHTED_SUM,
        },
        {
            'name': 'FidelityFocused',
            'weights': (0.20, 0.65, 0.15),
            'scalarization': ScalarizationMethod.WEIGHTED_SUM,
        },
        {
            'name': 'Balanced',
            'weights': (0.35, 0.45, 0.20),
            'scalarization': ScalarizationMethod.WEIGHTED_SUM,
        },
        {
            'name': 'Chebyshev',
            'weights': (0.33, 0.34, 0.33),
            'scalarization': ScalarizationMethod.CHEBYSHEV,
        },
        {
            'name': 'ConstraintBased',
            'weights': (0.40, 0.45, 0.15),
            'scalarization': ScalarizationMethod.CONSTRAINT_BASED,
        },
    ]
    
    mo_results = []
    mo_training_histories = {}
    
    for mo_cfg in mo_training_configs:
        print(f"\n--- Training {mo_cfg['name']} Agent ---")
        print(f"    Weights: time={mo_cfg['weights'][0]}, fidelity={mo_cfg['weights'][1]}, energy={mo_cfg['weights'][2]}")
        
        # Create environment with THIS objective configuration
        mo_config = MultiObjectiveConfig(
            weight_time=mo_cfg['weights'][0],
            weight_fidelity=mo_cfg['weights'][1],
            weight_energy=mo_cfg['weights'][2],
            scalarization=mo_cfg['scalarization'],
            min_fidelity_threshold=0.10
        )
        
        env_mo = FedMOEnv(
            dataset_path=config.dataset_path,
            n_qnodes=config.n_qnodes,
            n_qtasks_per_episode=config.n_qtasks_per_episode,
            mo_config=mo_config,
            seed=config.seed + hash(mo_cfg['name']) % 1000
        )
        
        # Train a NEW agent for THIS configuration
        agent_mo = RainbowDQNAgent(
            state_dim=env_mo.observation_space.shape[0],
            action_dim=config.n_qnodes,
            config=rainbow_config,
            device=DEVICE
        )
        
        # Full training
        history, trained_agent = run_drl_training(
            env_mo, agent_mo,
            n_episodes=config.n_episodes,
            eval_interval=100,
            verbose=False
        )
        
        mo_training_histories[mo_cfg['name']] = history
        
        # Compute final metrics (average of last 100 episodes)
        final_fidelity = np.mean([h.get('mean_fidelity', 0) for h in history[-100:]])
        final_time = np.mean([h.get('mean_completion_time', 0) for h in history[-100:]])
        final_energy = np.mean([h.get('total_energy', 0) for h in history[-100:]])
        final_reward = np.mean([h.get('mean_reward', 0) for h in history[-100:]])
        
        print(f"    Results: Fidelity={final_fidelity:.4f}, Time={final_time:.1f}s, Reward={final_reward:.4f}")
        
        # Record for each episode (for learning curves)
        for ep, h in enumerate(history):
            mo_results.append({
                'episode': ep,
                'config_name': mo_cfg['name'],
                'scalarization': mo_cfg['scalarization'].value,
                'weight_time': mo_cfg['weights'][0],
                'weight_fidelity': mo_cfg['weights'][1],
                'weight_energy': mo_cfg['weights'][2],
                'mean_fidelity': h.get('mean_fidelity', 0),
                'mean_completion_time': h.get('mean_completion_time', 0),
                'total_energy': h.get('total_energy', 0),
                'mean_reward': h.get('mean_reward', 0)
            })
    
    # Save detailed results
    mo_df = pd.DataFrame(mo_results)
    mo_df.to_csv(os.path.join(output_dir, "multi_objective_results.csv"), index=False)
    
    # Generate summary table
    print("\n--- Multi-Objective Summary ---")
    summary_data = []
    for mo_cfg in mo_training_configs:
        cfg_data = mo_df[mo_df['config_name'] == mo_cfg['name']]
        final_data = cfg_data.tail(100)  # Last 100 episodes
        summary_data.append({
            'Configuration': mo_cfg['name'],
            'Fidelity': f"{final_data['mean_fidelity'].mean():.4f}",
            'Time': f"{final_data['mean_completion_time'].mean():.1f}",
            'Reward': f"{final_data['mean_reward'].mean():.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(output_dir, "multi_objective_summary.csv"), index=False)

    # ========== EXPERIMENT 4: Ablation Study ==========
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Ablation Study - Error-Aware Features")
    print("=" * 60)
    
    # Without error features
    print("\nTraining WITHOUT error-aware features...")
    
    env_no_error = FedMOEnv(
        dataset_path=config.dataset_path,
        n_qnodes=config.n_qnodes,
        n_qtasks_per_episode=config.n_qtasks_per_episode,
        include_error_features=False, # Disable error features
        seed=config.seed + 2
    )
    
    agent_no_error = RainbowDQNAgent(
        state_dim=env_no_error.observation_space.shape[0],
        action_dim=config.n_qnodes,
        config=rainbow_config,
        device=DEVICE
    )
    
    history_no_error, _ = run_drl_training(
        env_no_error, agent_no_error, config.n_episodes,
        eval_interval=20, verbose=True
    )
    
    # Compare
    ablation_df = pd.DataFrame(history_no_error)
    ablation_df['variant'] = 'NoErrorFeatures'
    
    drl_df_copy = drl_df.copy()
    drl_df_copy['variant'] = 'WithErrorFeatures'
    
    ablation_combined = pd.concat([drl_df_copy, ablation_df])
    ablation_combined.to_csv(os.path.join(output_dir, "ablation_results.csv"), index=False)

    # ========== EXPERIMENT 5: Federated Learning ==========
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Federated Learning Comparison")
    print("=" * 60)
    
    # PATCH 4: First train centralized baseline (same compute budget)
    print("\n--- Centralized Training (Baseline) ---")
    
    total_federated_episodes = config.federated_rounds * config.local_steps
    
    centralized_env = FedMOEnv(
        dataset_path=config.dataset_path,
        n_qnodes=config.n_qnodes,
        n_qtasks_per_episode=config.n_qtasks_per_episode,
        seed=config.seed
    )
    
    centralized_agent = RainbowDQNAgent(
        state_dim=centralized_env.observation_space.shape[0],
        action_dim=config.n_qnodes,
        config=rainbow_config,
        device=DEVICE
    )
    
    centralized_history, _ = run_drl_training(
        centralized_env, centralized_agent,
        n_episodes=total_federated_episodes,
        eval_interval=50,
        verbose=False
    )
    
    centralized_final = {
        'method': 'Centralized',
        'fed_round': -1,
        'mean_fidelity': np.mean([h.get('mean_fidelity', 0) for h in centralized_history[-20:]]),
        'mean_completion_time': np.mean([h.get('mean_completion_time', 0) for h in centralized_history[-20:]]),
        'communication_cost': 0,
        'privacy': 'None'
    }
    
    print(f"Centralized: Fidelity={centralized_final['mean_fidelity']:.4f}, "
          f"Time={centralized_final['mean_completion_time']:.1f}s")
    
    fed_configs = {
        'FedAvg': AggregationMethod.FEDAVG,
        'FedProx': AggregationMethod.FEDPROX,
        'FedNova': AggregationMethod.FEDNOVA
    }
    
    fed_results = [centralized_final]  # Start with centralized baseline
    
    for name, method in fed_configs.items():
        print(f"\nTraining with {name}...")
        
        fed_config = FederatedConfig(
            aggregation=method,
            local_steps=config.local_steps,
            num_rounds=config.federated_rounds,
            fedprox_mu=0.1  # RESEARCH: Increased for non-IID quantum workloads
        )
        
        aggregator = FederatedAggregator(
            config=fed_config,
            n_clients=config.n_datacenters,
            device=DEVICE
        )
        
        # Create agents for each datacenter
        agents = []
        envs = []
        
        for dc_id in range(config.n_datacenters):
            env_dc = FedMOEnv(
                dataset_path=config.dataset_path,
                n_qnodes=config.n_qnodes,
                n_qtasks_per_episode=config.n_qtasks_per_episode,
                datacenter_id=dc_id,
                seed=config.seed + dc_id * 100
            )
            envs.append(env_dc)
            
            agent_dc = RainbowDQNAgent(
                state_dim=env_dc.observation_space.shape[0],
                action_dim=config.n_qnodes,
                config=rainbow_config,
                device=DEVICE
            )
            agents.append(agent_dc)
            
            # Register with aggregator
            aggregator.register_client(
                client_id=dc_id,
                datacenter_name=f"DC_{dc_id}",
                initial_params=agent_dc.get_parameters()
            )
            
        # Federated training rounds
        round_metrics = []
        
        for fed_round in range(config.federated_rounds):
            # Local training on each datacenter
            for dc_id in range(config.n_datacenters):
                env = envs[dc_id]
                agent = agents[dc_id]
                
                # Sync with global model
                global_params = aggregator.get_global_model()
                agent.set_parameters(global_params)
                
                # Local training steps
                for _ in range(config.local_steps):
                    obs, _ = env.reset()
                    done = False
                    step_count = 0
                    max_steps = 200  # Prevent infinite loops
                    while not done and step_count < max_steps:
                        action_mask = env.get_action_mask() if hasattr(env, 'get_action_mask') else None
                        action = agent.select_action(obs, training=True, action_mask=action_mask)
                        next_obs, reward, done, truncated, info = env.step(action)
                        agent.store_transition(obs, action, reward, next_obs, done)
                        agent.update()
                        obs = next_obs
                        step_count += 1
                        if truncated: done = True
                        
                # Submit update
                aggregator.submit_update(
                    client_id=dc_id,
                    updated_params=agent.get_parameters(),
                    local_steps=config.local_steps * config.n_qtasks_per_episode, # approx
                    local_samples=config.local_steps * config.n_qtasks_per_episode
                )
            
            # Aggregate
            aggregator.aggregate()
            
            # Evaluate using global model on client 0's env
            global_params = aggregator.get_global_model()
            eval_agent = RainbowDQNAgent(
                state_dim=envs[0].observation_space.shape[0],
                action_dim=config.n_qnodes,
                config=rainbow_config,
                device=DEVICE
            )
            eval_agent.set_parameters(global_params)
            
            # Quick eval
            obs, _ = envs[0].reset()
            done = False
            eval_steps = 0
            max_eval_steps = 200  # Prevent infinite loops
            while not done and eval_steps < max_eval_steps:
                action_mask = envs[0].get_action_mask() if hasattr(envs[0], 'get_action_mask') else None
                action = eval_agent.select_action(obs, training=False, action_mask=action_mask)
                obs, _, done, truncated, _ = envs[0].step(action)
                eval_steps += 1
                if truncated: done = True
                
            summary = envs[0].get_episode_summary()
            summary['fed_round'] = fed_round
            summary['method'] = name
            round_metrics.append(summary)
            
            if (fed_round + 1) % 5 == 0:
                print(f"  Round {fed_round+1}: Fidelity={summary.get('mean_fidelity', 0):.4f}, "
                      f"Time={summary.get('mean_completion_time', 0):.2f}s")
                      
        fed_results.extend(round_metrics)
        
    fed_df = pd.DataFrame(fed_results)
    fed_df.to_csv(os.path.join(output_dir, "federated_results.csv"), index=False)

    # ========== Generate Summary ==========
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    summary = {
        'timestamp': timestamp,
        'dataset': dataset_name,
        'config': asdict(config),
        'heuristics': {
            name: {
                'mean_reward': heuristics_df[heuristics_df['policy'] == name]['mean_reward'].mean(),
                'mean_fidelity': heuristics_df[heuristics_df['policy'] == name]['mean_fidelity'].mean(),
                'mean_completion_time': heuristics_df[heuristics_df['policy'] == name]['mean_completion_time'].mean()
            }
            for name in heuristic_policies.keys()
        },
        'drl': {
            'final_mean_reward': drl_df['mean_reward'].tail(20).mean(),
            'final_mean_fidelity': drl_df['mean_fidelity'].tail(20).mean(),
            'final_mean_completion_time': drl_df['mean_completion_time'].tail(20).mean()
        }
    }
    
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"\nResults saved to: {output_dir}")
    return output_dir

if __name__ == "__main__":
    # Define the dataset directory relative to this script
    # Assumes script is in experiments/ and qsimpy is parallel to experiments/
    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qsimpy/qdataset")
    
    # Find all CSV files in the dataset directory
    dataset_files = glob.glob(os.path.join(dataset_dir, "qsimpyds_*.csv"))
    
    # Sort files to ensure reproducible order
    dataset_files.sort()
    
    print(f"Found {len(dataset_files)} datasets to process.")
    
    if not dataset_files:
        print("WARNING: No datasets found! Checking default path...")
        # Fallback for direct execution
        dataset_files = glob.glob("../qsimpy/qdataset/qsimpyds_*.csv")
        print(f"Found {len(dataset_files)} datasets in fallback path.")

    for i, dataset_file in enumerate(dataset_files):
        print(f"\n\n{'#'*80}")
        print(f"PROCESSING DATASET {i+1}/{len(dataset_files)}: {os.path.basename(dataset_file)}")
        print(f"{'#'*80}\n")
        
        config = ExperimentConfig(
            dataset_path=dataset_file,
            n_qnodes=5,
            n_qtasks_per_episode=25,
            n_episodes=500,  # Increased for Rainbow DQN convergence
            n_eval_episodes=20,
            n_datacenters=3,
            federated_rounds=10,
            local_steps=5,
            seed=42
        )
        
        try:
            run_experiments(config)
            print(f"\n[SUCCESS] Finished dataset: {dataset_file}")
        except Exception as e:
            print(f"\n[ERROR] Failed to train on {dataset_file}")
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()
