"""
FedMO-DRLQ Utilities Module
===========================
Helper functions and utilities for the FedMO-DRLQ framework.

Author: Sandhya (NIT Sikkim)
"""

import numpy as np
import torch
import random
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the best available device"""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float = 0.005):
    """Soft update target network parameters"""
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)


def hard_update(target: torch.nn.Module, source: torch.nn.Module):
    """Hard update (copy) target network parameters"""
    target.load_state_dict(source.state_dict())


class RunningMeanStd:
    """Running mean and standard deviation calculator"""
    
    def __init__(self, shape: Tuple = (), epsilon: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x: np.ndarray):
        """Update running statistics with new batch"""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        
        new_var = m2 / total_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = total_count
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize input using running statistics"""
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


class ExponentialMovingAverage:
    """Exponential moving average tracker"""
    
    def __init__(self, alpha: float = 0.99):
        self.alpha = alpha
        self.value = None
    
    def update(self, new_value: float):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * new_value
    
    def get(self) -> Optional[float]:
        return self.value


class MetricsTracker:
    """Track and aggregate training metrics"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics: Dict[str, List[float]] = {}
    
    def add(self, name: str, value: float):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
        
        # Keep only recent values
        if len(self.metrics[name]) > self.window_size:
            self.metrics[name] = self.metrics[name][-self.window_size:]
    
    def get_mean(self, name: str) -> float:
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        return np.mean(self.metrics[name])
    
    def get_std(self, name: str) -> float:
        if name not in self.metrics or len(self.metrics[name]) < 2:
            return 0.0
        return np.std(self.metrics[name])
    
    def get_all_means(self) -> Dict[str, float]:
        return {name: self.get_mean(name) for name in self.metrics}
    
    def reset(self):
        self.metrics = {}


def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
    """Create directory for experiment with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def save_json(data: Dict, path: str):
    """Save dictionary to JSON file"""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str) -> Dict:
    """Load dictionary from JSON file"""
    with open(path, 'r') as f:
        return json.load(f)


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_number(n: float) -> str:
    """Format large numbers with K/M/B suffixes"""
    if abs(n) >= 1e9:
        return f"{n/1e9:.1f}B"
    elif abs(n) >= 1e6:
        return f"{n/1e6:.1f}M"
    elif abs(n) >= 1e3:
        return f"{n/1e3:.1f}K"
    else:
        return f"{n:.1f}"


class LinearSchedule:
    """Linear schedule for hyperparameter annealing"""
    
    def __init__(self, start: float, end: float, steps: int):
        self.start = start
        self.end = end
        self.steps = steps
        self.current_step = 0
    
    def step(self) -> float:
        progress = min(self.current_step / self.steps, 1.0)
        value = self.start + progress * (self.end - self.start)
        self.current_step += 1
        return value
    
    def reset(self):
        self.current_step = 0


class CosineSchedule:
    """Cosine annealing schedule"""
    
    def __init__(self, start: float, end: float, steps: int):
        self.start = start
        self.end = end
        self.steps = steps
        self.current_step = 0
    
    def step(self) -> float:
        progress = min(self.current_step / self.steps, 1.0)
        cosine = 0.5 * (1 + np.cos(np.pi * progress))
        value = self.end + (self.start - self.end) * cosine
        self.current_step += 1
        return value
    
    def reset(self):
        self.current_step = 0


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    last_value: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalized Advantage Estimation.
    
    Args:
        rewards: List of rewards
        values: List of value estimates
        dones: List of done flags
        gamma: Discount factor
        gae_lambda: GAE lambda
        last_value: Value estimate for final state
    
    Returns:
        Tuple of (advantages, returns)
    """
    n = len(rewards)
    advantages = np.zeros(n)
    returns = np.zeros(n)
    
    gae = 0
    values_extended = list(values) + [last_value]
    
    for t in reversed(range(n)):
        if dones[t]:
            delta = rewards[t] - values[t]
            gae = delta
        else:
            delta = rewards[t] + gamma * values_extended[t + 1] - values[t]
            gae = delta + gamma * gae_lambda * gae
        
        advantages[t] = gae
        returns[t] = gae + values[t]
    
    return advantages, returns


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute explained variance for value function evaluation"""
    var_y = np.var(y_true)
    if var_y == 0:
        return 0.0
    return 1 - np.var(y_true - y_pred) / var_y


def normalize_advantages(advantages: np.ndarray) -> np.ndarray:
    """Normalize advantages to zero mean and unit variance"""
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
