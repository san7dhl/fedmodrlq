"""
Evaluation Module for FedMO-DRLQ
================================
Comprehensive evaluation framework for benchmarking FedMO-DRLQ
against baseline algorithms.

Baselines:
- Heuristic: FCFS, SJF, Min-Min, Random, Greedy-Error
- DRL: Standard DQN, Double DQN, Dueling DQN, PPO
- Multi-Objective: NSGA-II weighted sum

Metrics:
- Completion Time (makespan, average, total)
- Fidelity (average, minimum, distribution)
- Energy Consumption
- Pareto Optimality
- Scalability

Author: Sandhya (NIT Sikkim)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
import json
from pathlib import Path

from ..core import (
    FedMODRLQConfig,
    QNodeErrorMetrics,
    MultiObjectiveRewardFunction,
    RewardComponents
)


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode"""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    
    # Time metrics
    total_completion_time: float = 0.0
    avg_completion_time: float = 0.0
    max_completion_time: float = 0.0
    min_completion_time: float = float('inf')
    makespan: float = 0.0
    
    # Fidelity metrics
    avg_fidelity: float = 0.0
    min_fidelity: float = float('inf')
    max_fidelity: float = 0.0
    fidelity_std: float = 0.0
    
    # Energy metrics
    total_energy: float = 0.0
    avg_energy: float = 0.0
    
    # Reward metrics
    total_reward: float = 0.0
    avg_reward: float = 0.0
    
    # Scheduling metrics
    total_rescheduling: int = 0
    avg_queue_length: float = 0.0
    node_utilization: Dict[int, float] = field(default_factory=dict)
    
    # Timing
    episode_duration: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "total_completion_time": self.total_completion_time,
            "avg_completion_time": self.avg_completion_time,
            "max_completion_time": self.max_completion_time,
            "min_completion_time": self.min_completion_time if self.min_completion_time != float('inf') else 0,
            "makespan": self.makespan,
            "avg_fidelity": self.avg_fidelity,
            "min_fidelity": self.min_fidelity if self.min_fidelity != float('inf') else 0,
            "max_fidelity": self.max_fidelity,
            "fidelity_std": self.fidelity_std,
            "total_energy": self.total_energy,
            "avg_energy": self.avg_energy,
            "total_reward": self.total_reward,
            "avg_reward": self.avg_reward,
            "total_rescheduling": self.total_rescheduling,
            "node_utilization": self.node_utilization,
            "episode_duration": self.episode_duration
        }


@dataclass
class BenchmarkResults:
    """Results from benchmarking an algorithm"""
    algorithm_name: str
    episodes: List[EpisodeMetrics] = field(default_factory=list)
    training_time: float = 0.0
    inference_time: float = 0.0
    
    # Aggregated metrics
    mean_completion_time: float = 0.0
    std_completion_time: float = 0.0
    mean_fidelity: float = 0.0
    std_fidelity: float = 0.0
    mean_energy: float = 0.0
    std_energy: float = 0.0
    mean_reward: float = 0.0
    std_reward: float = 0.0
    
    # Pareto metrics
    pareto_solutions: List[Tuple[float, float, float]] = field(default_factory=list)
    hypervolume: float = 0.0
    
    def compute_aggregates(self):
        """Compute aggregate statistics from episodes"""
        if not self.episodes:
            return
        
        completion_times = [e.avg_completion_time for e in self.episodes]
        fidelities = [e.avg_fidelity for e in self.episodes]
        energies = [e.avg_energy for e in self.episodes]
        rewards = [e.avg_reward for e in self.episodes]
        
        self.mean_completion_time = np.mean(completion_times)
        self.std_completion_time = np.std(completion_times)
        self.mean_fidelity = np.mean(fidelities)
        self.std_fidelity = np.std(fidelities)
        self.mean_energy = np.mean(energies)
        self.std_energy = np.std(energies)
        self.mean_reward = np.mean(rewards)
        self.std_reward = np.std(rewards)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm_name": self.algorithm_name,
            "num_episodes": len(self.episodes),
            "training_time": self.training_time,
            "inference_time": self.inference_time,
            "mean_completion_time": self.mean_completion_time,
            "std_completion_time": self.std_completion_time,
            "mean_fidelity": self.mean_fidelity,
            "std_fidelity": self.std_fidelity,
            "mean_energy": self.mean_energy,
            "std_energy": self.std_energy,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "hypervolume": self.hypervolume
        }


# =============================================================================
# Baseline Algorithms
# =============================================================================

class BaselineScheduler(ABC):
    """Abstract base class for baseline scheduling algorithms"""
    
    @abstractmethod
    def select_action(self, observation: np.ndarray, valid_actions: List[int]) -> int:
        """Select action given observation"""
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return algorithm name"""
        pass
    
    def reset(self):
        """Reset scheduler state"""
        pass


class FCFSScheduler(BaselineScheduler):
    """First-Come-First-Served Scheduler"""
    
    def __init__(self, n_qnodes: int = 5):
        self.n_qnodes = n_qnodes
        self.current_node = 0
    
    def select_action(self, observation: np.ndarray, valid_actions: List[int]) -> int:
        # Round-robin assignment
        action = self.current_node % self.n_qnodes
        self.current_node += 1
        
        # Ensure valid action
        if action not in valid_actions and valid_actions:
            action = valid_actions[0]
        
        return action
    
    def name(self) -> str:
        return "FCFS"
    
    def reset(self):
        self.current_node = 0


class SJFScheduler(BaselineScheduler):
    """Shortest Job First (assigns to node with shortest queue)"""
    
    def __init__(self, n_qnodes: int = 5, obs_qnode_start: int = 4, obs_qnode_dim: int = 10):
        self.n_qnodes = n_qnodes
        self.obs_qnode_start = obs_qnode_start
        self.obs_qnode_dim = obs_qnode_dim
    
    def select_action(self, observation: np.ndarray, valid_actions: List[int]) -> int:
        # Extract next_available_time for each node (index 2 in node features)
        best_action = valid_actions[0] if valid_actions else 0
        best_time = float('inf')
        
        for action in valid_actions:
            node_start = self.obs_qnode_start + action * self.obs_qnode_dim
            next_available = observation[node_start + 2]  # next_available_time index
            
            if next_available < best_time:
                best_time = next_available
                best_action = action
        
        return best_action
    
    def name(self) -> str:
        return "SJF"


class MinMinScheduler(BaselineScheduler):
    """Min-Min Scheduler (assigns to node with minimum completion time)"""
    
    def __init__(self, n_qnodes: int = 5, obs_qnode_start: int = 4, obs_qnode_dim: int = 10):
        self.n_qnodes = n_qnodes
        self.obs_qnode_start = obs_qnode_start
        self.obs_qnode_dim = obs_qnode_dim
    
    def select_action(self, observation: np.ndarray, valid_actions: List[int]) -> int:
        # Extract task info
        task_qubits = observation[1]
        task_depth = observation[2]
        
        best_action = valid_actions[0] if valid_actions else 0
        best_completion = float('inf')
        
        for action in valid_actions:
            node_start = self.obs_qnode_start + action * self.obs_qnode_dim
            node_qubits = observation[node_start]
            node_clops = observation[node_start + 1]
            next_available = observation[node_start + 2]
            
            # Check qubit constraint
            if task_qubits > node_qubits:
                continue
            
            # Estimate execution time
            exec_time = task_depth / max(node_clops, 1) * 1024  # Assuming 1024 shots
            completion_time = max(0, next_available) + exec_time
            
            if completion_time < best_completion:
                best_completion = completion_time
                best_action = action
        
        return best_action
    
    def name(self) -> str:
        return "Min-Min"


class RandomScheduler(BaselineScheduler):
    """Random Scheduler"""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
    
    def select_action(self, observation: np.ndarray, valid_actions: List[int]) -> int:
        return self.rng.choice(valid_actions) if valid_actions else 0
    
    def name(self) -> str:
        return "Random"


class GreedyErrorScheduler(BaselineScheduler):
    """Greedy Error-Aware Scheduler (selects node with lowest error rate)"""
    
    def __init__(self, n_qnodes: int = 5, obs_qnode_start: int = 4, obs_qnode_dim: int = 10):
        self.n_qnodes = n_qnodes
        self.obs_qnode_start = obs_qnode_start
        self.obs_qnode_dim = obs_qnode_dim
    
    def select_action(self, observation: np.ndarray, valid_actions: List[int]) -> int:
        # In error-aware observation, index 3 is two_qubit_error (normalized)
        best_action = valid_actions[0] if valid_actions else 0
        best_error = float('inf')
        
        for action in valid_actions:
            node_start = self.obs_qnode_start + action * self.obs_qnode_dim
            
            # Check if error features are included
            if self.obs_qnode_dim >= 10:
                two_qubit_error = observation[node_start + 3]  # Normalized error
            else:
                two_qubit_error = 0.5  # Default if not available
            
            if two_qubit_error < best_error:
                best_error = two_qubit_error
                best_action = action
        
        return best_action
    
    def name(self) -> str:
        return "Greedy-Error"


class FidelityAwareScheduler(BaselineScheduler):
    """Fidelity-Aware Scheduler (selects node with highest estimated fidelity)"""
    
    def __init__(self, n_qnodes: int = 5, obs_qnode_start: int = 4, obs_qnode_dim: int = 10):
        self.n_qnodes = n_qnodes
        self.obs_qnode_start = obs_qnode_start
        self.obs_qnode_dim = obs_qnode_dim
    
    def select_action(self, observation: np.ndarray, valid_actions: List[int]) -> int:
        # In error-aware observation, index 9 is estimated_fidelity
        best_action = valid_actions[0] if valid_actions else 0
        best_fidelity = -1
        
        for action in valid_actions:
            node_start = self.obs_qnode_start + action * self.obs_qnode_dim
            
            if self.obs_qnode_dim >= 10:
                fidelity = observation[node_start + 9]  # Estimated fidelity
            else:
                fidelity = 0.5
            
            if fidelity > best_fidelity:
                best_fidelity = fidelity
                best_action = action
        
        return best_action
    
    def name(self) -> str:
        return "Fidelity-Aware"


# =============================================================================
# Metrics Calculation
# =============================================================================

class MetricsCalculator:
    """Calculate evaluation metrics from episode results"""
    
    def __init__(self):
        pass
    
    def calculate_episode_metrics(
        self,
        results: List[Dict[str, Any]],
        episode_duration: float = 0.0
    ) -> EpisodeMetrics:
        """
        Calculate metrics from episode results.
        
        Args:
            results: List of task result dictionaries
            episode_duration: Wall-clock time for episode
        
        Returns:
            EpisodeMetrics object
        """
        if not results:
            return EpisodeMetrics()
        
        metrics = EpisodeMetrics()
        metrics.total_tasks = len(results)
        metrics.completed_tasks = len(results)
        metrics.episode_duration = episode_duration
        
        # Extract values
        completion_times = [r['completion_time'] for r in results]
        fidelities = [r['fidelity'] for r in results]
        energies = [r['energy'] for r in results]
        rewards = [r['reward'] for r in results]
        reschedules = [r.get('rescheduling_count', 0) for r in results]
        
        # Time metrics
        metrics.total_completion_time = sum(completion_times)
        metrics.avg_completion_time = np.mean(completion_times)
        metrics.max_completion_time = max(completion_times)
        metrics.min_completion_time = min(completion_times)
        metrics.makespan = max(completion_times)  # Simplified
        
        # Fidelity metrics
        metrics.avg_fidelity = np.mean(fidelities)
        metrics.min_fidelity = min(fidelities)
        metrics.max_fidelity = max(fidelities)
        metrics.fidelity_std = np.std(fidelities)
        
        # Energy metrics
        metrics.total_energy = sum(energies)
        metrics.avg_energy = np.mean(energies)
        
        # Reward metrics
        metrics.total_reward = sum(rewards)
        metrics.avg_reward = np.mean(rewards)
        
        # Scheduling metrics
        metrics.total_rescheduling = sum(reschedules)
        
        # Node utilization
        node_counts: Dict[int, int] = {}
        for r in results:
            node_id = r.get('qnode_id', 0)
            node_counts[node_id] = node_counts.get(node_id, 0) + 1
        
        for node_id, count in node_counts.items():
            metrics.node_utilization[node_id] = count / len(results)
        
        return metrics
    
    def calculate_pareto_front(
        self,
        solutions: List[Tuple[float, float, float]]
    ) -> List[Tuple[float, float, float]]:
        """
        Calculate Pareto-optimal solutions.
        
        Args:
            solutions: List of (time, fidelity, energy) tuples
                      Note: Lower time and energy are better, higher fidelity is better
        
        Returns:
            List of Pareto-optimal solutions
        """
        if not solutions:
            return []
        
        pareto_front = []
        
        for solution in solutions:
            time, fidelity, energy = solution
            is_dominated = False
            
            for other in solutions:
                other_time, other_fidelity, other_energy = other
                
                # Check if other dominates solution
                # (lower time, higher fidelity, lower energy)
                if (other_time <= time and 
                    other_fidelity >= fidelity and 
                    other_energy <= energy and
                    (other_time < time or other_fidelity > fidelity or other_energy < energy)):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(solution)
        
        return pareto_front
    
    def calculate_hypervolume(
        self,
        pareto_front: List[Tuple[float, float, float]],
        reference_point: Tuple[float, float, float] = (100.0, 0.0, 100.0)
    ) -> float:
        """
        Calculate hypervolume indicator for Pareto front.
        
        Simplified 3D hypervolume calculation.
        
        Args:
            pareto_front: List of Pareto-optimal solutions
            reference_point: Reference point (worst case for each objective)
        
        Returns:
            Hypervolume value
        """
        if not pareto_front:
            return 0.0
        
        # Normalize solutions
        ref_time, ref_fidelity, ref_energy = reference_point
        
        # Simple approximation using sum of dominated volumes
        hypervolume = 0.0
        
        for time, fidelity, energy in pareto_front:
            # Volume contribution (simplified)
            if time < ref_time and fidelity > ref_fidelity and energy < ref_energy:
                vol = (ref_time - time) * (fidelity - ref_fidelity) * (ref_energy - energy)
                hypervolume += vol
        
        return hypervolume


# =============================================================================
# Benchmark Runner
# =============================================================================

class BenchmarkRunner:
    """
    Run benchmarks comparing FedMO-DRLQ against baselines.
    """
    
    def __init__(
        self,
        env_factory: Callable[[], Any],
        metrics_calculator: Optional[MetricsCalculator] = None
    ):
        """
        Initialize benchmark runner.
        
        Args:
            env_factory: Function that creates environment instances
            metrics_calculator: Optional custom metrics calculator
        """
        self.env_factory = env_factory
        self.metrics = metrics_calculator or MetricsCalculator()
        
        # Results storage
        self.results: Dict[str, BenchmarkResults] = {}
    
    def benchmark_baseline(
        self,
        scheduler: BaselineScheduler,
        num_episodes: int = 10,
        max_steps: int = 1000
    ) -> BenchmarkResults:
        """
        Benchmark a baseline scheduler.
        
        Args:
            scheduler: Baseline scheduler to evaluate
            num_episodes: Number of episodes to run
            max_steps: Maximum steps per episode
        
        Returns:
            BenchmarkResults object
        """
        results = BenchmarkResults(algorithm_name=scheduler.name())
        
        total_inference_time = 0.0
        
        for episode in range(num_episodes):
            env = self.env_factory()
            scheduler.reset()
            
            obs, info = env.reset()
            episode_start = time.time()
            
            for step in range(max_steps):
                # Get valid actions (all actions for now)
                valid_actions = list(range(env.action_space.n))
                
                # Select action
                inference_start = time.time()
                action = scheduler.select_action(obs, valid_actions)
                total_inference_time += time.time() - inference_start
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    break
            
            episode_duration = time.time() - episode_start
            
            # Calculate metrics
            episode_results = env.episode_results if hasattr(env, 'episode_results') else []
            episode_metrics = self.metrics.calculate_episode_metrics(
                episode_results, episode_duration
            )
            results.episodes.append(episode_metrics)
            
            env.close()
        
        results.inference_time = total_inference_time
        results.compute_aggregates()
        
        return results
    
    def benchmark_agent(
        self,
        agent: Any,
        agent_name: str,
        num_episodes: int = 10,
        max_steps: int = 1000,
        training: bool = False
    ) -> BenchmarkResults:
        """
        Benchmark a DRL agent.
        
        Args:
            agent: DRL agent with select_action method
            agent_name: Name for the agent
            num_episodes: Number of episodes
            max_steps: Maximum steps per episode
            training: Whether agent is training (affects exploration)
        
        Returns:
            BenchmarkResults object
        """
        results = BenchmarkResults(algorithm_name=agent_name)
        
        total_inference_time = 0.0
        
        for episode in range(num_episodes):
            env = self.env_factory()
            obs, info = env.reset()
            episode_start = time.time()
            
            for step in range(max_steps):
                # Select action
                inference_start = time.time()
                action = agent.select_action(obs, training=training)
                total_inference_time += time.time() - inference_start
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    break
            
            episode_duration = time.time() - episode_start
            
            # Calculate metrics
            episode_results = env.episode_results if hasattr(env, 'episode_results') else []
            episode_metrics = self.metrics.calculate_episode_metrics(
                episode_results, episode_duration
            )
            results.episodes.append(episode_metrics)
            
            env.close()
        
        results.inference_time = total_inference_time
        results.compute_aggregates()
        
        return results
    
    def run_all_baselines(self, num_episodes: int = 10) -> Dict[str, BenchmarkResults]:
        """Run all baseline algorithms"""
        
        # Create baselines
        baselines = [
            FCFSScheduler(n_qnodes=5),
            SJFScheduler(n_qnodes=5),
            MinMinScheduler(n_qnodes=5),
            RandomScheduler(),
            GreedyErrorScheduler(n_qnodes=5),
            FidelityAwareScheduler(n_qnodes=5)
        ]
        
        results = {}
        for scheduler in baselines:
            print(f"Benchmarking {scheduler.name()}...")
            results[scheduler.name()] = self.benchmark_baseline(
                scheduler, num_episodes=num_episodes
            )
        
        self.results.update(results)
        return results
    
    def compare_results(self) -> Dict[str, Any]:
        """
        Compare all benchmark results.
        
        Returns:
            Comparison summary
        """
        if not self.results:
            return {}
        
        comparison = {
            "algorithms": list(self.results.keys()),
            "completion_time": {},
            "fidelity": {},
            "energy": {},
            "reward": {}
        }
        
        for name, result in self.results.items():
            comparison["completion_time"][name] = {
                "mean": result.mean_completion_time,
                "std": result.std_completion_time
            }
            comparison["fidelity"][name] = {
                "mean": result.mean_fidelity,
                "std": result.std_fidelity
            }
            comparison["energy"][name] = {
                "mean": result.mean_energy,
                "std": result.std_energy
            }
            comparison["reward"][name] = {
                "mean": result.mean_reward,
                "std": result.std_reward
            }
        
        # Find best algorithms
        comparison["best"] = {
            "completion_time": min(self.results.items(), 
                                   key=lambda x: x[1].mean_completion_time)[0],
            "fidelity": max(self.results.items(), 
                           key=lambda x: x[1].mean_fidelity)[0],
            "energy": min(self.results.items(), 
                         key=lambda x: x[1].mean_energy)[0],
            "reward": max(self.results.items(), 
                         key=lambda x: x[1].mean_reward)[0]
        }
        
        return comparison
    
    def save_results(self, path: str):
        """Save benchmark results to JSON file"""
        output = {
            name: result.to_dict() 
            for name, result in self.results.items()
        }
        
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
    
    def print_summary(self):
        """Print summary of benchmark results"""
        comparison = self.compare_results()
        
        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 70)
        
        # Header
        print(f"\n{'Algorithm':<20} {'Time (avg)':<15} {'Fidelity (avg)':<15} "
              f"{'Energy (avg)':<15} {'Reward (avg)':<15}")
        print("-" * 80)
        
        for name in comparison["algorithms"]:
            time_val = comparison["completion_time"][name]["mean"]
            fid_val = comparison["fidelity"][name]["mean"]
            energy_val = comparison["energy"][name]["mean"]
            reward_val = comparison["reward"][name]["mean"]
            
            print(f"{name:<20} {time_val:<15.4f} {fid_val:<15.4f} "
                  f"{energy_val:<15.6f} {reward_val:<15.4f}")
        
        print("-" * 80)
        print(f"\nBest Algorithms:")
        print(f"  Completion Time: {comparison['best']['completion_time']}")
        print(f"  Fidelity: {comparison['best']['fidelity']}")
        print(f"  Energy: {comparison['best']['energy']}")
        print(f"  Reward: {comparison['best']['reward']}")
        print("=" * 70)
