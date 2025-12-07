"""
Multi-Objective Reward Module for FedMO-DRLQ
=============================================
Implements multi-objective reward functions for quantum cloud scheduling:
- Completion time optimization
- Fidelity maximization
- Energy consumption minimization

Supports three scalarization strategies:
1. Weighted Sum
2. Chebyshev Scalarization
3. Constraint-Based Optimization

This addresses Gap 2: MORL Has Never Been Applied to Quantum Cloud Scheduling

Author: Sandhya (NIT Sikkim)
Reference: FedMO-DRLQ Research Gap Analysis
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import warnings


class ObjectiveType(Enum):
    """Types of optimization objectives"""
    COMPLETION_TIME = "completion_time"
    FIDELITY = "fidelity"
    ENERGY = "energy"


class ScalarizationMethod(Enum):
    """Multi-objective scalarization strategies"""
    WEIGHTED_SUM = "weighted_sum"
    CHEBYSHEV = "chebyshev"
    CONSTRAINT_BASED = "constraint_based"


@dataclass
class ObjectiveConfig:
    """Configuration for a single objective"""
    name: ObjectiveType
    weight: float
    is_minimization: bool  # True for costs, False for benefits
    scale_factor: float = 1.0
    constraint_threshold: Optional[float] = None


@dataclass
class MultiObjectiveConfig:
    """Configuration for multi-objective optimization
    
    RESEARCH: Updated thresholds based on actual NISQ fidelity ranges.
    """
    
    # Weights for objectives (must sum to 1.0)
    # UPDATED: Increased fidelity weight to improve DRL learning signal
    weight_time: float = 0.25
    weight_fidelity: float = 0.6  # Highest priority for fidelity optimization
    weight_energy: float = 0.15
    
    # Scalarization method
    scalarization: ScalarizationMethod = ScalarizationMethod.WEIGHTED_SUM
    
    # Constraint thresholds (for constraint-based scalarization)
    # RESEARCH: Fixed based on actual data - mean fidelity is 0.11-0.15
    min_fidelity_threshold: float = 0.10  # Was 0.8 - unrealistic for NISQ
    max_time_threshold: float = 30000.0  # Reasonable time constraint
    max_energy_threshold: float = 2.0e9  # FIXED: Match actual energy range
    
    # CRITICAL FIX: Ideal/Nadir bounds based on ACTUAL experimental data
    # Previous bounds caused flat reward landscape (everything clipped to extremes)
    # Time bounds (seconds) - actual data range: 2,417 - 36,929
    ideal_time: float = 2000.0      # Best achievable ~2400s
    nadir_time: float = 40000.0     # Worst observed ~37000s
    
    # Fidelity bounds - actual data range: 0.10 - 0.20
    ideal_fidelity: float = 0.25    # Target for good NISQ scheduling
    nadir_fidelity: float = 0.05    # Worst acceptable
    
    # Energy bounds (Joules) - actual data range: 2.8e8 - 1.6e9
    ideal_energy: float = 2.0e8     # Best observed ~2.8e8
    nadir_energy: float = 2.0e9     # Worst observed ~1.6e9
    
    # Reward scaling factors
    time_scale: float = 1.0
    fidelity_scale: float = 1.0  # All objectives on same scale now
    energy_scale: float = 1.0
    
    # Fidelity bonus reward shaping
    fidelity_bonus_threshold: float = 0.12  # Fidelity threshold for bonus
    fidelity_bonus_value: float = 0.5  # Bonus reward when threshold exceeded
    
    def validate_weights(self) -> bool:
        """Validate that weights sum to 1.0"""
        total = self.weight_time + self.weight_fidelity + self.weight_energy
        return np.isclose(total, 1.0, atol=1e-6)
    
    def get_weights(self) -> Tuple[float, float, float]:
        """Return weights as tuple"""
        return (self.weight_time, self.weight_fidelity, self.weight_energy)


class MultiObjectiveReward:
    """
    Multi-Objective Reward Calculator for Quantum Cloud Scheduling.
    
    Computes rewards based on three objectives:
    1. Completion Time (minimize)
    2. Fidelity (maximize)
    3. Energy (minimize)
    
    Supports multiple scalarization strategies for combining objectives.
    """
    
    def __init__(self, config: Optional[MultiObjectiveConfig] = None):
        """
        Initialize multi-objective reward calculator.
        
        Args:
            config: Multi-objective configuration
        """
        self.config = config or MultiObjectiveConfig()
        
        if not self.config.validate_weights():
            warnings.warn("Objective weights do not sum to 1.0, normalizing...")
            self._normalize_weights()
        
        # History for Pareto analysis
        self.objective_history: List[Dict[str, float]] = []
        self.reward_history: List[float] = []
    
    def compute_step_reward(
        self,
        is_valid_placement: bool,
        expected_fidelity: float,
        node_utilization: float,
        queue_length: int,
        max_queue_length: int = 10
    ) -> float:
        """
        Compute immediate step reward for dense reward shaping.
        
        This provides intermediate rewards for each scheduling decision,
        helping the agent learn faster with sparse episode rewards.
        
        Args:
            is_valid_placement: Whether action was a valid node placement
            expected_fidelity: Estimated fidelity for this task-node pair
            node_utilization: Current utilization ratio of selected node (0-1)
            queue_length: Current queue length at selected node
            max_queue_length: Maximum expected queue length
            
        Returns:
            Immediate step reward (dense feedback)
        """
        if not is_valid_placement:
            return -1.0  # Strong penalty for invalid actions
        
        reward = 0.0
        
        # 1. Fidelity-aware placement bonus (immediate feedback)
        # Scale fidelity to meaningful range based on NISQ reality
        fidelity_normalized = (expected_fidelity - 0.05) / (0.25 - 0.05)  # 0 to 1 range
        fidelity_normalized = np.clip(fidelity_normalized, 0.0, 1.0)
        reward += 0.5 * fidelity_normalized
        
        # 2. Load balancing reward (sweet spot: 30-80% utilization)
        if 0.3 < node_utilization < 0.8:
            reward += 0.1  # Prefer balanced load
        elif node_utilization > 0.95:
            reward -= 0.1  # Penalize overloaded nodes
        
        # 3. Queue management reward (prefer shorter queues)
        queue_ratio = queue_length / max_queue_length
        if queue_ratio < 0.3:
            reward += 0.05
        elif queue_ratio > 0.8:
            reward -= 0.05
        
        return reward
        
    def _normalize_weights(self):
        """Normalize weights to sum to 1.0"""
        total = self.config.weight_time + self.config.weight_fidelity + self.config.weight_energy
        if total > 0:
            self.config.weight_time /= total
            self.config.weight_fidelity /= total
            self.config.weight_energy /= total
    
    def normalize_objective(
        self,
        value: float,
        ideal: float,
        nadir: float,
        is_minimization: bool = True
    ) -> float:
        """
        Normalize objective value to [0, 1] range.
        
        For minimization objectives: 0 is ideal, 1 is nadir
        For maximization objectives: 1 is ideal, 0 is nadir
        
        Args:
            value: Raw objective value
            ideal: Ideal (best) value
            nadir: Nadir (worst) value
            is_minimization: Whether objective should be minimized
            
        Returns:
            Normalized value in [0, 1]
        """
        if nadir == ideal:
            return 0.5  # Avoid division by zero
        
        if is_minimization:
            # Lower is better: normalize so 0 is ideal, 1 is nadir
            normalized = (value - ideal) / (nadir - ideal)
        else:
            # Higher is better: normalize so 1 is ideal, 0 is nadir
            normalized = (value - nadir) / (ideal - nadir)
        
        return np.clip(normalized, 0.0, 1.0)
    
    def compute_individual_rewards(
        self,
        completion_time: float,
        fidelity: float,
        energy: float
    ) -> Dict[str, float]:
        """
        Compute individual reward components for each objective.
        
        Args:
            completion_time: Task completion time (seconds)
            fidelity: Estimated circuit fidelity [0, 1]
            energy: Energy consumption (Joules)
            
        Returns:
            Dictionary of normalized rewards for each objective
        """
        # Normalize objectives
        r_time = 1.0 - self.normalize_objective(
            completion_time,
            self.config.ideal_time,
            self.config.nadir_time,
            is_minimization=True
        )
        
        r_fidelity = self.normalize_objective(
            fidelity,
            self.config.ideal_fidelity,
            self.config.nadir_fidelity,
            is_minimization=False
        )
        
        r_energy = 1.0 - self.normalize_objective(
            energy,
            self.config.ideal_energy,
            self.config.nadir_energy,
            is_minimization=True
        )
        
        return {
            'time': r_time * self.config.time_scale,
            'fidelity': r_fidelity * self.config.fidelity_scale,
            'energy': r_energy * self.config.energy_scale
        }
    
    def weighted_sum(
        self,
        completion_time: float,
        fidelity: float,
        energy: float
    ) -> float:
        """
        Compute reward using weighted sum scalarization.
        
        R = w_time * R_time + w_fidelity * R_fidelity + w_energy * R_energy
        
        Args:
            completion_time: Task completion time (seconds)
            fidelity: Estimated circuit fidelity [0, 1]
            energy: Energy consumption (Joules)
            
        Returns:
            Scalar reward value
        """
        rewards = self.compute_individual_rewards(completion_time, fidelity, energy)
        
        total_reward = (
            self.config.weight_time * rewards['time'] +
            self.config.weight_fidelity * rewards['fidelity'] +
            self.config.weight_energy * rewards['energy']
        )
        
        # Apply fidelity bonus reward shaping
        if fidelity >= self.config.fidelity_bonus_threshold:
            total_reward += self.config.fidelity_bonus_value
        
        return total_reward
    
    def chebyshev(
        self,
        completion_time: float,
        fidelity: float,
        energy: float
    ) -> float:
        """
        Compute reward using Chebyshev scalarization.
        
        This method can find solutions in non-convex Pareto regions.
        Minimizes the maximum weighted deviation from the ideal point.
        
        R = -max(w_i * |f_i - ideal_i|)  (negated for maximization)
        
        Args:
            completion_time: Task completion time (seconds)
            fidelity: Estimated circuit fidelity [0, 1]
            energy: Energy consumption (Joules)
            
        Returns:
            Scalar reward value
        """
        # Normalize objectives to [0, 1]
        norm_time = self.normalize_objective(
            completion_time,
            self.config.ideal_time,
            self.config.nadir_time,
            is_minimization=True
        )
        
        norm_fidelity = 1.0 - self.normalize_objective(
            fidelity,
            self.config.ideal_fidelity,
            self.config.nadir_fidelity,
            is_minimization=False
        )
        
        norm_energy = self.normalize_objective(
            energy,
            self.config.ideal_energy,
            self.config.nadir_energy,
            is_minimization=True
        )
        
        # Compute weighted deviations from ideal (ideal is 0 after normalization for minimization)
        deviations = [
            self.config.weight_time * norm_time,
            self.config.weight_fidelity * norm_fidelity,
            self.config.weight_energy * norm_energy
        ]
        
        # Return negative of max deviation (to convert minimization to maximization)
        return -max(deviations) + 1.0  # Shift to positive range
    
    def constraint_based(
        self,
        completion_time: float,
        fidelity: float,
        energy: float
    ) -> float:
        """
        Compute reward using constraint-based optimization.
        
        Optimizes primary objective (completion time) while enforcing
        constraints on secondary objectives (fidelity and energy).
        
        Applies penalties when constraints are violated.
        
        Args:
            completion_time: Task completion time (seconds)
            fidelity: Estimated circuit fidelity [0, 1]
            energy: Energy consumption (Joules)
            
        Returns:
            Scalar reward value
        """
        # Primary reward: time
        rewards = self.compute_individual_rewards(completion_time, fidelity, energy)
        primary_reward = rewards['time']
        
        # Constraint penalties
        penalty = 0.0
        
        # Fidelity constraint
        if fidelity < self.config.min_fidelity_threshold:
            fidelity_violation = self.config.min_fidelity_threshold - fidelity
            penalty += 10.0 * fidelity_violation  # Strong penalty
        
        # Energy constraint
        if energy > self.config.max_energy_threshold:
            energy_violation = (energy - self.config.max_energy_threshold) / self.config.max_energy_threshold
            penalty += 5.0 * energy_violation
        
        # Final reward
        return primary_reward - penalty
    
    def compute_reward(
        self,
        completion_time: float,
        fidelity: float,
        energy: float
    ) -> float:
        """
        Compute total reward using configured scalarization method.
        
        Args:
            completion_time: Task completion time (seconds)
            fidelity: Estimated circuit fidelity [0, 1]
            energy: Energy consumption (Joules)
            
        Returns:
            Scalar reward value
        """
        # Store for Pareto analysis
        self.objective_history.append({
            'completion_time': completion_time,
            'fidelity': fidelity,
            'energy': energy
        })
        
        # Compute reward based on method
        if self.config.scalarization == ScalarizationMethod.WEIGHTED_SUM:
            reward = self.weighted_sum(completion_time, fidelity, energy)
        elif self.config.scalarization == ScalarizationMethod.CHEBYSHEV:
            reward = self.chebyshev(completion_time, fidelity, energy)
        elif self.config.scalarization == ScalarizationMethod.CONSTRAINT_BASED:
            reward = self.constraint_based(completion_time, fidelity, energy)
        else:
            reward = self.weighted_sum(completion_time, fidelity, energy)
        
        self.reward_history.append(reward)
        return reward
    
    def get_pareto_front(self) -> List[Dict[str, float]]:
        """
        Extract Pareto-optimal solutions from history.
        
        Returns:
            List of non-dominated solutions
        """
        if not self.objective_history:
            return []
        
        pareto_front = []
        
        for i, sol_i in enumerate(self.objective_history):
            is_dominated = False
            
            for j, sol_j in enumerate(self.objective_history):
                if i == j:
                    continue
                
                # Check if sol_j dominates sol_i
                # Dominance: better or equal in all objectives, strictly better in at least one
                time_better = sol_j['completion_time'] <= sol_i['completion_time']
                fidelity_better = sol_j['fidelity'] >= sol_i['fidelity']
                energy_better = sol_j['energy'] <= sol_i['energy']
                
                all_better_or_equal = time_better and fidelity_better and energy_better
                
                strictly_better = (
                    sol_j['completion_time'] < sol_i['completion_time'] or
                    sol_j['fidelity'] > sol_i['fidelity'] or
                    sol_j['energy'] < sol_i['energy']
                )
                
                if all_better_or_equal and strictly_better:
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(sol_i)
        
        return pareto_front
    
    def get_hypervolume(self, reference_point: Optional[Dict[str, float]] = None) -> float:
        """
        Compute hypervolume indicator for Pareto quality assessment.
        
        Args:
            reference_point: Reference point (nadir) for hypervolume calculation
            
        Returns:
            Hypervolume value
        """
        pareto_front = self.get_pareto_front()
        
        if not pareto_front:
            return 0.0
        
        # Default reference point
        if reference_point is None:
            reference_point = {
                'completion_time': self.config.nadir_time,
                'fidelity': self.config.nadir_fidelity,
                'energy': self.config.nadir_energy
            }
        
        # Simplified 3D hypervolume calculation (exact for small fronts)
        # For larger fronts, use WFG or other efficient algorithms
        hypervolume = 0.0
        
        for sol in pareto_front:
            # Volume of dominated region for this solution
            vol = (
                max(0, reference_point['completion_time'] - sol['completion_time']) *
                max(0, sol['fidelity'] - reference_point['fidelity']) *
                max(0, reference_point['energy'] - sol['energy'])
            )
            hypervolume += vol
        
        return hypervolume
    
    def reset_history(self):
        """Reset objective and reward history"""
        self.objective_history = []
        self.reward_history = []
    
    def get_statistics(self) -> Dict[str, float]:
        """Get statistics on reward computation"""
        if not self.reward_history:
            return {}
        
        return {
            'mean_reward': np.mean(self.reward_history),
            'std_reward': np.std(self.reward_history),
            'min_reward': np.min(self.reward_history),
            'max_reward': np.max(self.reward_history),
            'n_samples': len(self.reward_history),
            'pareto_front_size': len(self.get_pareto_front())
        }


# Convenience functions for creating pre-configured reward calculators
def create_time_focused_reward() -> MultiObjectiveReward:
    """Create reward calculator focused on completion time"""
    config = MultiObjectiveConfig(
        weight_time=0.7,
        weight_fidelity=0.2,
        weight_energy=0.1,
        scalarization=ScalarizationMethod.WEIGHTED_SUM
    )
    return MultiObjectiveReward(config)


def create_fidelity_focused_reward() -> MultiObjectiveReward:
    """Create reward calculator focused on fidelity"""
    config = MultiObjectiveConfig(
        weight_time=0.2,
        weight_fidelity=0.7,
        weight_energy=0.1,
        scalarization=ScalarizationMethod.WEIGHTED_SUM
    )
    return MultiObjectiveReward(config)


def create_balanced_reward() -> MultiObjectiveReward:
    """Create reward calculator with fidelity-prioritized objectives"""
    config = MultiObjectiveConfig(
        weight_time=0.25,
        weight_fidelity=0.6,  # Prioritize fidelity for DRL learning
        weight_energy=0.15,
        scalarization=ScalarizationMethod.WEIGHTED_SUM
    )
    return MultiObjectiveReward(config)


def create_chebyshev_reward() -> MultiObjectiveReward:
    """Create reward calculator using Chebyshev scalarization"""
    config = MultiObjectiveConfig(
        weight_time=0.25,
        weight_fidelity=0.6,  # Prioritize fidelity
        weight_energy=0.15,
        scalarization=ScalarizationMethod.CHEBYSHEV
    )
    return MultiObjectiveReward(config)


def create_constraint_reward(
    min_fidelity: float = 0.8,
    max_energy: float = 100000.0
) -> MultiObjectiveReward:
    """Create reward calculator with constraint-based optimization"""
    config = MultiObjectiveConfig(
        weight_time=0.8,
        weight_fidelity=0.1,
        weight_energy=0.1,
        scalarization=ScalarizationMethod.CONSTRAINT_BASED,
        min_fidelity_threshold=min_fidelity,
        max_energy_threshold=max_energy
    )
    return MultiObjectiveReward(config)