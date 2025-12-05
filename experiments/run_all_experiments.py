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
from fedmo_drlq.federated.federated_learning import (
    FederatedAggregator, FederatedTrainer, FederatedConfig, AggregationMethod
)
# CRITICAL FIX: Import the real environment instead of defining SimplifiedEnv
from fedmo_drlq.envs.fedmo_env import FedMOEnv

@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    # Dataset
    dataset_path: str
    
    # Environment
    n_qnodes: int = 5
    n_qtasks_per_episode: int = 25
    
    # Training
    n_episodes: int = 200
    n_eval_episodes: int = 50
    
    # Federated
    n_datacenters: int = 3
    federated_rounds: int = 20
    local_steps: int = 10
    
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
            action = agent.select_action(obs, training=True)
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Merge done and truncated for training signal
            training_done = done or truncated
            
            agent.store_transition(obs, action, reward, next_obs, training_done)
            loss = agent.update()
            
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
        lr=0.0001,
        gamma=0.99,
        batch_size=64,
        buffer_size=50000,
        min_buffer_size=500,
        target_update_freq=500,
        use_double_dqn=True,
        use_prioritized_replay=True,
        use_dueling=True,
        use_multistep=True,
        n_step=3,
        use_distributional=True,
        num_atoms=51,
        use_noisy_nets=True
    )
    
    agent = RainbowDQNAgent(
        state_dim=env_drl.observation_space.shape[0],
        action_dim=config.n_qnodes,
        config=rainbow_config,
        device=DEVICE
    )
    
    print("\nTraining Rainbow DQN with error-aware features...")
    drl_history, trained_agent = run_drl_training(
        env_drl, agent, config.n_episodes, 
        eval_interval=20, verbose=True
    )
    
    drl_df = pd.DataFrame(drl_history)
    drl_df['policy'] = 'RainbowDQN_ErrorAware'
    drl_df.to_csv(os.path.join(output_dir, "rainbow_dqn_results.csv"), index=False)
    
    # Save trained agent
    trained_agent.save(os.path.join(output_dir, "rainbow_dqn_agent.pt"))
    
    # ========== EXPERIMENT 3: Multi-Objective Comparison ==========
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Multi-Objective Scalarization Comparison")
    print("=" * 60)
    
    scalarization_methods = {
        'WeightedSum': ScalarizationMethod.WEIGHTED_SUM,
        'Chebyshev': ScalarizationMethod.CHEBYSHEV,
        'ConstraintBased': ScalarizationMethod.CONSTRAINT_BASED
    }
    
    mo_results = []
    
    for name, method in scalarization_methods.items():
        print(f"\nTraining with {name} scalarization...")
        
        mo_config = MultiObjectiveConfig(
            weight_time=0.4,
            weight_fidelity=0.4,
            weight_energy=0.2,
            scalarization=method,
            min_fidelity_threshold=0.7
        )
        
        env_mo = FedMOEnv(
            dataset_path=config.dataset_path,
            n_qnodes=config.n_qnodes,
            n_qtasks_per_episode=config.n_qtasks_per_episode,
            mo_config=mo_config,
            seed=config.seed + 1
        )
        
        agent_mo = RainbowDQNAgent(
            state_dim=env_mo.observation_space.shape[0],
            action_dim=config.n_qnodes,
            config=rainbow_config,
            device=DEVICE
        )
        
        history, _ = run_drl_training(
            env_mo, agent_mo, config.n_episodes // 2, # Shorter run for MO comparison
            eval_interval=20, verbose=False
        )
        
        for h in history:
            h['scalarization'] = name
            mo_results.append(h)
            
        mean_fidelity = np.mean([h.get('mean_fidelity', 0) for h in history[-20:]]) if history else 0
        mean_time = np.mean([h.get('mean_completion_time', 0) for h in history[-20:]]) if history else 0
        print(f"  Final Mean Fidelity: {mean_fidelity:.4f}")
        print(f"  Final Mean Completion Time: {mean_time:.2f}s")
        
    mo_df = pd.DataFrame(mo_results)
    mo_df.to_csv(os.path.join(output_dir, "multi_objective_results.csv"), index=False)

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
    
    fed_configs = {
        'FedAvg': AggregationMethod.FEDAVG,
        'FedProx': AggregationMethod.FEDPROX,
        'FedNova': AggregationMethod.FEDNOVA
    }
    
    fed_results = []
    
    for name, method in fed_configs.items():
        print(f"\nTraining with {name}...")
        
        fed_config = FederatedConfig(
            aggregation=method,
            local_steps=config.local_steps,
            num_rounds=config.federated_rounds,
            fedprox_mu=0.01
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
                        action = agent.select_action(obs, training=True)
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
                action = eval_agent.select_action(obs, training=False)
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
            n_episodes=100,
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
