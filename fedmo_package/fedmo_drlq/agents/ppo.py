"""
PPO Agent for FedMO-DRLQ
========================
Proximal Policy Optimization agent for quantum cloud scheduling.

PPO is a policy gradient method that provides:
- Stable training through clipped objective
- Good sample efficiency
- Natural handling of continuous and discrete actions
- Better suited for multi-objective scenarios

Author: Sandhya (NIT Sikkim)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

try:
    from ..core.config import TrainingConfig
except ImportError:
    from dataclasses import dataclass
    @dataclass
    class TrainingConfig:
        algorithm: str = "ppo"
        gamma: float = 0.99
        lr: float = 0.001
        ppo_clip: float = 0.2
        ppo_epochs: int = 10
        batch_size: int = 64


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic Network for PPO.
    
    Shares feature extraction layers between actor (policy) and critic (value).
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
        
        # Smaller init for output layers
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            Tuple of (action_logits, value)
        """
        features = self.shared(state)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """
        Get action from policy.
        
        Returns:
            Tuple of (action, log_prob, value)
        """
        action_logits, value = self.forward(state)
        
        if deterministic:
            action = action_logits.argmax(dim=-1)
            dist = Categorical(logits=action_logits)
            log_prob = dist.log_prob(action)
        else:
            dist = Categorical(logits=action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob, value.squeeze(-1)
    
    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.
        
        Returns:
            Tuple of (log_probs, values, entropy)
        """
        action_logits, values = self.forward(states)
        dist = Categorical(logits=action_logits)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values.squeeze(-1), entropy


@dataclass
class RolloutBuffer:
    """Buffer for storing rollout data"""
    states: List[np.ndarray]
    actions: List[int]
    rewards: List[float]
    values: List[float]
    log_probs: List[float]
    dones: List[bool]
    
    def __init__(self):
        self.clear()
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def __len__(self) -> int:
        return len(self.states)


class PPOAgent:
    """
    Proximal Policy Optimization Agent.
    
    Features:
    - Clipped surrogate objective for stable updates
    - Generalized Advantage Estimation (GAE)
    - Entropy bonus for exploration
    - Support for federated learning
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[TrainingConfig] = None,
        device: str = "cpu"
    ):
        """
        Initialize PPO agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            config: Training configuration
            device: Device to use
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or TrainingConfig(algorithm="ppo")
        self.device = torch.device(device)
        
        # Hyperparameters
        self.gamma = self.config.gamma
        self.lr = self.config.lr
        self.clip_ratio = self.config.ppo_clip
        self.n_epochs = self.config.ppo_epochs
        self.batch_size = self.config.batch_size
        self.gae_lambda = 0.95
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.max_grad_norm = 0.5
        
        # Network
        self.network = ActorCriticNetwork(state_dim, action_dim).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
        
        # Rollout buffer
        self.buffer = RolloutBuffer()
        
        # Metrics
        self.policy_losses = []
        self.value_losses = []
        self.entropy_values = []
        self.update_count = 0
    
    def select_action(
        self,
        state: np.ndarray,
        training: bool = True
    ) -> Tuple[int, float, float]:
        """
        Select action using current policy.
        
        Returns:
            Tuple of (action, log_prob, value)
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.network.get_action(
                state_t, 
                deterministic=not training
            )
        
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Store transition in rollout buffer"""
        self.buffer.add(state, action, reward, value, log_prob, done)
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        last_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.
        
        Returns:
            Tuple of (advantages, returns)
        """
        n_steps = len(rewards)
        advantages = np.zeros(n_steps, dtype=np.float32)
        returns = np.zeros(n_steps, dtype=np.float32)
        
        # Add last value for bootstrapping
        values_extended = values + [last_value]
        
        gae = 0
        for t in reversed(range(n_steps)):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * values_extended[t + 1] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
            
            advantages[t] = gae
            returns[t] = gae + values[t]
        
        return advantages, returns
    
    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        """
        Perform PPO update.
        
        Args:
            last_value: Value estimate for the last state (for bootstrapping)
        
        Returns:
            Dictionary of loss values
        """
        if len(self.buffer) < self.batch_size:
            return {}
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(
            self.buffer.rewards,
            self.buffer.values,
            self.buffer.dones,
            last_value
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        actions = torch.LongTensor(self.buffer.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        
        # Training loop
        n_samples = len(self.buffer)
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        for _ in range(self.n_epochs):
            # Random permutation for mini-batches
            indices = np.random.permutation(n_samples)
            
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Get batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages_t[batch_indices]
                batch_returns = returns_t[batch_indices]
                
                # Evaluate actions
                new_log_probs, values, entropy = self.network.evaluate_actions(
                    batch_states, batch_actions
                )
                
                # Policy loss (clipped surrogate)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss + 
                    self.value_coef * value_loss + 
                    self.entropy_coef * entropy_loss
                )
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1
        
        # Clear buffer
        self.buffer.clear()
        self.update_count += 1
        
        # Average losses
        avg_policy_loss = total_policy_loss / n_updates
        avg_value_loss = total_value_loss / n_updates
        avg_entropy = total_entropy / n_updates
        
        self.policy_losses.append(avg_policy_loss)
        self.value_losses.append(avg_value_loss)
        self.entropy_values.append(avg_entropy)
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy
        }
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get network parameters for federated learning"""
        return {k: v.cpu().clone() for k, v in self.network.state_dict().items()}
    
    def set_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Set network parameters from federated aggregation"""
        self.network.load_state_dict(parameters)
    
    def save(self, path: str):
        """Save agent state"""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'update_count': self.update_count,
            'config': self.config
        }, path)
    
    def load(self, path: str):
        """Load agent state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.update_count = checkpoint['update_count']
    
    def get_metrics(self) -> Dict[str, float]:
        """Get training metrics"""
        return {
            'avg_policy_loss': np.mean(self.policy_losses[-100:]) if self.policy_losses else 0,
            'avg_value_loss': np.mean(self.value_losses[-100:]) if self.value_losses else 0,
            'avg_entropy': np.mean(self.entropy_values[-100:]) if self.entropy_values else 0,
            'update_count': self.update_count
        }
    
    def reset_metrics(self):
        """Reset training metrics"""
        self.policy_losses = []
        self.value_losses = []
        self.entropy_values = []
