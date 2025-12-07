"""
Evaluation Module for FedMO-DRLQ
================================
Comprehensive evaluation framework for benchmarking FedMO-DRLQ against
baseline algorithms and analyzing multi-objective performance.

Features:
- Baseline algorithm implementations (heuristics)
- Pareto front analysis
- Statistical comparison utilities
- Visualization tools

Author: Sandhya (NIT Sikkim)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import json
from abc import ABC, abstractmethod


# =============================================================================
# Baseline Schedulers (Heuristics)
# =============================================================================

class BaselineScheduler(ABC):
    """Abstract base class for baseline scheduling algorithms"""
    
    @abstractmethod
    def select_action(self, observation: np.ndarray, valid_actions: List[int]) -> int:
        """Select action based on current observation"""
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return scheduler name"""
        pass


class FCFSScheduler(BaselineScheduler):
    """
    First-Come-First-Served Scheduler
    
    Assigns tasks to the first available node that can handle them.
    """
    
    def select_action(self, observation: np.ndarray, valid_actions: List[int]) -> int:
        # Simply return first valid action
        return valid_actions[0] if valid_actions else 0
    
    def name(self) -> str:
        return "FCFS"


class SJFScheduler(BaselineScheduler):
    """
    Shortest Job First Scheduler
    
    Assigns tasks to the node with shortest estimated completion time.
    """
    
    def __init__(self, n_qnodes: int = 5, qnode_feature_dim: int = 10):
        self.n_qnodes = n_qnodes
        self.qnode_feature_dim = qnode_feature_dim
    
    def select_action(self, observation: np.ndarray, valid_actions: List[int]) -> int:
        # Extract task features (first 4 elements)
        task_layers = observation[2]  # circuit_layers
        
        # Extract node features
        qtask_dim = 4
        best_action = valid_actions[0]
        best_time = float('inf')
        
        for action in valid_actions:
            node_start = qtask_dim + action * self.qnode_feature_dim
            clops = observation[node_start + 1]
            next_available = observation[node_start + 2]
            
            # Estimate execution time
            exec_time = task_layers / max(clops, 1)
            total_time = max(0, next_available) + exec_time
            
            if total_time < best_time:
                best_time = total_time
                best_action = action
        
        return best_action
    
    def name(self) -> str:
        return "SJF"


class MinCompletionTimeScheduler(BaselineScheduler):
    """
    Minimum Completion Time (MCT) Scheduler
    
    Assigns tasks to minimize total completion time.
    """
    
    def __init__(self, n_qnodes: int = 5, qnode_feature_dim: int = 10):
        self.n_qnodes = n_qnodes
        self.qnode_feature_dim = qnode_feature_dim
    
    def select_action(self, observation: np.ndarray, valid_actions: List[int]) -> int:
        qtask_dim = 4
        task_layers = observation[2]
        task_qubits = observation[1]
        
        best_action = valid_actions[0]
        min_completion = float('inf')
        
        for action in valid_actions:
            node_start = qtask_dim + action * self.qnode_feature_dim
            node_qubits = observation[node_start]
            clops = observation[node_start + 1]
            next_available = observation[node_start + 2]
            
            # Check constraint
            if task_qubits > node_qubits:
                continue
            
            exec_time = task_layers / max(clops, 1)
            completion = max(0, next_available) + exec_time
            
            if completion < min_completion:
                min_completion = completion
                best_action = action
        
        return best_action
    
    def name(self) -> str:
        return "MCT"


class RandomScheduler(BaselineScheduler):
    """Random Scheduler - selects randomly among valid nodes"""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
    
    def select_action(self, observation: np.ndarray, valid_actions: List[int]) -> int:
        return self.rng.choice(valid_actions)
    
    def name(self) -> str:
        return "Random"


class RoundRobinScheduler(BaselineScheduler):
    """Round Robin Scheduler - cycles through nodes"""
    
    def __init__(self, n_qnodes: int = 5):
        self.n_qnodes = n_qnodes
        self.current = 0
    
    def select_action(self, observation: np.ndarray, valid_actions: List[int]) -> int:
        # Find next valid action in round-robin order
        for _ in range(self.n_qnodes):
            if self.current in valid_actions:
                action = self.current
                self.current = (self.current + 1) % self.n_qnodes
                return action
            self.current = (self.current + 1) % self.n_qnodes
        
        return valid_actions[0]
    
    def name(self) -> str:
        return "RoundRobin"


class GreedyFidelityScheduler(BaselineScheduler):
    """
    Greedy Fidelity Scheduler
    
    Selects the node with lowest error rate (highest expected fidelity).
    This is an error-aware heuristic baseline.
    """
    
    def __init__(self, n_qnodes: int = 5, qnode_feature_dim: int = 10):
        self.n_qnodes = n_qnodes
        self.qnode_feature_dim = qnode_feature_dim
    
    def select_action(self, observation: np.ndarray, valid_actions: List[int]) -> int:
        qtask_dim = 4
        
        best_action = valid_actions[0]
        best_fidelity = -1
        
        for action in valid_actions:
            node_start = qtask_dim + action * self.qnode_feature_dim
            
            # Get fidelity estimate (last feature in error-aware representation)
            if self.qnode_feature_dim >= 10:
                fidelity = observation[node_start + 9]  # estimated_fidelity
            else:
                # Fallback: use inverse of next_available_time as proxy
                fidelity = 1.0 / max(observation[node_start + 2] + 1, 1)
            
            if fidelity > best_fidelity:
                best_fidelity = fidelity
                best_action = action
        
        return best_action
    
    def name(self) -> str:
        return "GreedyFidelity"


class BalancedScheduler(BaselineScheduler):
    """
    Balanced Scheduler
    
    Balances between completion time and fidelity using weighted scoring.
    """
    
    def __init__(self, n_qnodes: int = 5, qnode_feature_dim: int = 10,
                 time_weight: float = 0.5, fidelity_weight: float = 0.5):
        self.n_qnodes = n_qnodes
        self.qnode_feature_dim = qnode_feature_dim
        self.time_weight = time_weight
        self.fidelity_weight = fidelity_weight
    
    def select_action(self, observation: np.ndarray, valid_actions: List[int]) -> int:
        qtask_dim = 4
        task_layers = observation[2]
        
        best_action = valid_actions[0]
        best_score = float('-inf')
        
        for action in valid_actions:
            node_start = qtask_dim + action * self.qnode_feature_dim
            clops = observation[node_start + 1]
            next_available = observation[node_start + 2]
            
            # Time score (inverse of completion time)
            exec_time = task_layers / max(clops, 1)
            completion = max(0, next_available) + exec_time
            time_score = 1.0 / max(completion, 0.001)
            
            # Fidelity score
            if self.qnode_feature_dim >= 10:
                fidelity_score = observation[node_start + 9]
            else:
                fidelity_score = 0.5
            
            # Combined score
            score = self.time_weight * time_score + self.fidelity_weight * fidelity_score
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action
    
    def name(self) -> str:
        return f"Balanced(t={self.time_weight},f={self.fidelity_weight})"


# =============================================================================
# Pareto Analysis
# =============================================================================

@dataclass
class ParetoPoint:
    """A point in the objective space"""
    completion_time: float
    fidelity: float
    energy: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def dominates(self, other: 'ParetoPoint') -> bool:
        """Check if this point dominates another"""
        # For minimization of time and energy, maximization of fidelity
        better_or_equal = (
            self.completion_time <= other.completion_time and
            self.fidelity >= other.fidelity and
            self.energy <= other.energy
        )
        strictly_better = (
            self.completion_time < other.completion_time or
            self.fidelity > other.fidelity or
            self.energy < other.energy
        )
        return better_or_equal and strictly_better
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array [time, -fidelity, energy] for minimization"""
        return np.array([self.completion_time, -self.fidelity, self.energy])


class ParetoFrontAnalyzer:
    """
    Analyzes Pareto fronts for multi-objective optimization.
    
    Provides metrics for comparing different algorithms:
    - Hypervolume indicator
    - Spread/diversity
    - Convergence metrics
    """
    
    def __init__(self, reference_point: Optional[Tuple[float, float, float]] = None):
        """
        Args:
            reference_point: Reference point for hypervolume calculation
                            (max_time, min_fidelity, max_energy)
        """
        self.reference_point = reference_point or (1000.0, 0.0, 100.0)
    
    def extract_pareto_front(self, points: List[ParetoPoint]) -> List[ParetoPoint]:
        """Extract non-dominated points from a set"""
        pareto_front = []
        
        for point in points:
            is_dominated = False
            points_to_remove = []
            
            for existing in pareto_front:
                if existing.dominates(point):
                    is_dominated = True
                    break
                if point.dominates(existing):
                    points_to_remove.append(existing)
            
            if not is_dominated:
                pareto_front = [p for p in pareto_front if p not in points_to_remove]
                pareto_front.append(point)
        
        return pareto_front
    
    def compute_hypervolume(self, points: List[ParetoPoint]) -> float:
        """
        Compute hypervolume indicator.
        
        Higher hypervolume indicates better Pareto front quality.
        Uses a simplified 2D projection for efficiency.
        """
        if not points:
            return 0.0
        
        # Project to 2D (time vs fidelity) for simplicity
        ref_time, ref_fidelity, _ = self.reference_point
        
        # Sort by time
        sorted_points = sorted(points, key=lambda p: p.completion_time)
        
        hypervolume = 0.0
        prev_fidelity = ref_fidelity
        
        for point in sorted_points:
            if point.completion_time < ref_time and point.fidelity > ref_fidelity:
                width = ref_time - point.completion_time
                height = point.fidelity - prev_fidelity
                hypervolume += width * max(0, height)
                prev_fidelity = max(prev_fidelity, point.fidelity)
        
        return hypervolume
    
    def compute_spread(self, points: List[ParetoPoint]) -> float:
        """
        Compute spread/diversity of Pareto front.
        
        Higher spread indicates better coverage of objective space.
        """
        if len(points) < 2:
            return 0.0
        
        # Convert to arrays
        arrays = [p.to_array() for p in points]
        
        # Compute pairwise distances
        distances = []
        for i, p1 in enumerate(arrays):
            for p2 in arrays[i+1:]:
                distances.append(np.linalg.norm(p1 - p2))
        
        if not distances:
            return 0.0
        
        return np.std(distances)
    
    def compute_convergence(self, 
                           achieved: List[ParetoPoint],
                           reference: List[ParetoPoint]) -> float:
        """
        Compute convergence metric (distance to reference front).
        
        Lower values indicate better convergence.
        """
        if not achieved or not reference:
            return float('inf')
        
        # Generational distance
        total_dist = 0.0
        for point in achieved:
            min_dist = float('inf')
            for ref_point in reference:
                dist = np.linalg.norm(point.to_array() - ref_point.to_array())
                min_dist = min(min_dist, dist)
            total_dist += min_dist
        
        return total_dist / len(achieved)
    
    def analyze(self, points: List[ParetoPoint]) -> Dict[str, float]:
        """Complete Pareto front analysis"""
        pareto_front = self.extract_pareto_front(points)
        
        return {
            "num_points": len(points),
            "pareto_size": len(pareto_front),
            "hypervolume": self.compute_hypervolume(pareto_front),
            "spread": self.compute_spread(pareto_front),
            "avg_time": np.mean([p.completion_time for p in pareto_front]) if pareto_front else 0,
            "avg_fidelity": np.mean([p.fidelity for p in pareto_front]) if pareto_front else 0,
            "avg_energy": np.mean([p.energy for p in pareto_front]) if pareto_front else 0,
        }


# =============================================================================
# Evaluation Metrics
# =============================================================================

@dataclass
class EpisodeMetrics:
    """Metrics for a single episode"""
    total_reward: float = 0.0
    avg_reward: float = 0.0
    total_completion_time: float = 0.0
    avg_completion_time: float = 0.0
    total_fidelity: float = 0.0
    avg_fidelity: float = 0.0
    total_energy: float = 0.0
    avg_energy: float = 0.0
    num_tasks: int = 0
    num_rescheduled: int = 0
    success_rate: float = 1.0


class Evaluator:
    """
    Comprehensive evaluator for FedMO-DRLQ.
    
    Compares DRL agents against baselines and analyzes
    multi-objective performance.
    """
    
    def __init__(self, env_factory: Callable, n_eval_episodes: int = 10):
        """
        Args:
            env_factory: Factory function to create evaluation environments
            n_eval_episodes: Number of episodes for evaluation
        """
        self.env_factory = env_factory
        self.n_eval_episodes = n_eval_episodes
        self.pareto_analyzer = ParetoFrontAnalyzer()
        
        # Results storage
        self.results: Dict[str, List[EpisodeMetrics]] = defaultdict(list)
        self.pareto_points: Dict[str, List[ParetoPoint]] = defaultdict(list)
    
    def evaluate_agent(
        self,
        agent: Any,
        agent_name: str,
        deterministic: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate a DRL agent.
        
        Args:
            agent: Agent with select_action method
            agent_name: Name for logging
            deterministic: Whether to use deterministic actions
        
        Returns:
            Average metrics across episodes
        """
        episode_metrics = []
        
        for _ in range(self.n_eval_episodes):
            env = self.env_factory()
            obs, _ = env.reset()
            
            total_reward = 0
            done = False
            step_count = 0
            max_steps = 200  # Prevent infinite loops
            
            while not done and step_count < max_steps:
                if hasattr(agent, 'select_action'):
                    action = agent.select_action(obs, training=not deterministic)
                    if isinstance(action, tuple):
                        action = action[0]
                else:
                    action = agent(obs)
                
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step_count += 1
                done = terminated or truncated
            
            # Get episode summary
            summary = env.get_episode_summary()
            
            metrics = EpisodeMetrics(
                total_reward=total_reward,
                avg_reward=summary.get('avg_reward', 0),
                total_completion_time=summary.get('total_completion_time', 0),
                avg_completion_time=summary.get('avg_completion_time', 0),
                total_fidelity=summary.get('total_fidelity', 0),
                avg_fidelity=summary.get('avg_fidelity', 0),
                total_energy=summary.get('total_energy', 0),
                avg_energy=summary.get('avg_energy', 0),
                num_tasks=summary.get('total_tasks', 0),
                num_rescheduled=summary.get('total_rescheduling', 0)
            )
            
            episode_metrics.append(metrics)
            self.results[agent_name].append(metrics)
            
            # Add Pareto point
            self.pareto_points[agent_name].append(ParetoPoint(
                completion_time=metrics.avg_completion_time,
                fidelity=metrics.avg_fidelity,
                energy=metrics.avg_energy,
                metadata={'agent': agent_name}
            ))
            
            env.close()
        
        # Compute averages
        return self._aggregate_metrics(episode_metrics)
    
    def evaluate_baseline(
        self,
        scheduler: BaselineScheduler
    ) -> Dict[str, float]:
        """Evaluate a baseline scheduler"""
        episode_metrics = []
        
        for _ in range(self.n_eval_episodes):
            env = self.env_factory()
            obs, _ = env.reset()
            
            total_reward = 0
            done = False
            step_count = 0
            max_steps = 200  # Prevent infinite loops
            
            while not done and step_count < max_steps:
                valid_actions = list(range(env.action_space.n))
                action = scheduler.select_action(obs, valid_actions)
                
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step_count += 1
                done = terminated or truncated
            
            summary = env.get_episode_summary()
            
            metrics = EpisodeMetrics(
                total_reward=total_reward,
                avg_reward=summary.get('avg_reward', 0),
                total_completion_time=summary.get('total_completion_time', 0),
                avg_completion_time=summary.get('avg_completion_time', 0),
                total_fidelity=summary.get('total_fidelity', 0),
                avg_fidelity=summary.get('avg_fidelity', 0),
                total_energy=summary.get('total_energy', 0),
                avg_energy=summary.get('avg_energy', 0),
                num_tasks=summary.get('total_tasks', 0),
                num_rescheduled=summary.get('total_rescheduling', 0)
            )
            
            episode_metrics.append(metrics)
            self.results[scheduler.name()].append(metrics)
            
            self.pareto_points[scheduler.name()].append(ParetoPoint(
                completion_time=metrics.avg_completion_time,
                fidelity=metrics.avg_fidelity,
                energy=metrics.avg_energy,
                metadata={'scheduler': scheduler.name()}
            ))
            
            env.close()
        
        return self._aggregate_metrics(episode_metrics)
    
    def _aggregate_metrics(self, metrics: List[EpisodeMetrics]) -> Dict[str, float]:
        """Aggregate metrics across episodes"""
        if not metrics:
            return {}
        
        return {
            'avg_reward': np.mean([m.avg_reward for m in metrics]),
            'std_reward': np.std([m.avg_reward for m in metrics]),
            'avg_completion_time': np.mean([m.avg_completion_time for m in metrics]),
            'std_completion_time': np.std([m.avg_completion_time for m in metrics]),
            'avg_fidelity': np.mean([m.avg_fidelity for m in metrics]),
            'std_fidelity': np.std([m.avg_fidelity for m in metrics]),
            'avg_energy': np.mean([m.avg_energy for m in metrics]),
            'std_energy': np.std([m.avg_energy for m in metrics]),
            'avg_rescheduling': np.mean([m.num_rescheduled for m in metrics]),
        }
    
    def compare_all(self) -> Dict[str, Dict[str, float]]:
        """Compare all evaluated methods"""
        comparison = {}
        
        for name, metrics in self.results.items():
            comparison[name] = self._aggregate_metrics(metrics)
            
            # Add Pareto analysis
            pareto_analysis = self.pareto_analyzer.analyze(self.pareto_points[name])
            comparison[name].update({f'pareto_{k}': v for k, v in pareto_analysis.items()})
        
        return comparison
    
    def get_improvement_over_baseline(
        self,
        method_name: str,
        baseline_name: str = "FCFS"
    ) -> Dict[str, float]:
        """Calculate percentage improvement over baseline"""
        if method_name not in self.results or baseline_name not in self.results:
            return {}
        
        method_metrics = self._aggregate_metrics(self.results[method_name])
        baseline_metrics = self._aggregate_metrics(self.results[baseline_name])
        
        improvements = {}
        
        # Time: lower is better
        if baseline_metrics['avg_completion_time'] > 0:
            improvements['time_reduction'] = (
                (baseline_metrics['avg_completion_time'] - method_metrics['avg_completion_time']) /
                baseline_metrics['avg_completion_time'] * 100
            )
        
        # Fidelity: higher is better
        if baseline_metrics['avg_fidelity'] > 0:
            improvements['fidelity_improvement'] = (
                (method_metrics['avg_fidelity'] - baseline_metrics['avg_fidelity']) /
                baseline_metrics['avg_fidelity'] * 100
            )
        
        # Energy: lower is better
        if baseline_metrics['avg_energy'] > 0:
            improvements['energy_reduction'] = (
                (baseline_metrics['avg_energy'] - method_metrics['avg_energy']) /
                baseline_metrics['avg_energy'] * 100
            )
        
        # Reward: higher is better
        if baseline_metrics['avg_reward'] != 0:
            improvements['reward_improvement'] = (
                (method_metrics['avg_reward'] - baseline_metrics['avg_reward']) /
                abs(baseline_metrics['avg_reward']) * 100
            )
        
        return improvements
    
    def generate_report(self) -> str:
        """Generate evaluation report"""
        comparison = self.compare_all()
        
        report = ["=" * 60]
        report.append("FedMO-DRLQ Evaluation Report")
        report.append("=" * 60)
        report.append("")
        
        for name, metrics in comparison.items():
            report.append(f"\n{name}:")
            report.append("-" * 40)
            report.append(f"  Avg Reward:          {metrics.get('avg_reward', 0):.4f} ± {metrics.get('std_reward', 0):.4f}")
            report.append(f"  Avg Completion Time: {metrics.get('avg_completion_time', 0):.4f} ± {metrics.get('std_completion_time', 0):.4f}")
            report.append(f"  Avg Fidelity:        {metrics.get('avg_fidelity', 0):.4f} ± {metrics.get('std_fidelity', 0):.4f}")
            report.append(f"  Avg Energy:          {metrics.get('avg_energy', 0):.6f} ± {metrics.get('std_energy', 0):.6f}")
            report.append(f"  Pareto Size:         {metrics.get('pareto_pareto_size', 0)}")
            report.append(f"  Hypervolume:         {metrics.get('pareto_hypervolume', 0):.4f}")
        
        # Improvements over FCFS
        if "FCFS" in comparison:
            report.append("\n" + "=" * 60)
            report.append("Improvement over FCFS Baseline")
            report.append("=" * 60)
            
            for name in comparison.keys():
                if name != "FCFS":
                    improvements = self.get_improvement_over_baseline(name, "FCFS")
                    if improvements:
                        report.append(f"\n{name}:")
                        for metric, value in improvements.items():
                            report.append(f"  {metric}: {value:+.2f}%")
        
        return "\n".join(report)
    
    def save_results(self, path: str):
        """Save results to JSON file"""
        comparison = self.compare_all()
        
        # Convert to serializable format
        serializable = {}
        for name, metrics in comparison.items():
            serializable[name] = {k: float(v) for k, v in metrics.items()}
        
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)


def get_all_baselines(n_qnodes: int = 5, qnode_feature_dim: int = 10) -> List[BaselineScheduler]:
    """Get all baseline schedulers"""
    return [
        FCFSScheduler(),
        SJFScheduler(n_qnodes, qnode_feature_dim),
        MinCompletionTimeScheduler(n_qnodes, qnode_feature_dim),
        RandomScheduler(),
        RoundRobinScheduler(n_qnodes),
        GreedyFidelityScheduler(n_qnodes, qnode_feature_dim),
        BalancedScheduler(n_qnodes, qnode_feature_dim, 0.5, 0.5),
    ]
