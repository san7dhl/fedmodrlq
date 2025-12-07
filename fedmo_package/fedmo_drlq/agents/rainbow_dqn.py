"""
Rainbow DQN Agent for FedMO-DRLQ
=================================
Rainbow DQN combines six improvements to DQN:
1. Double DQN - Reduces overestimation bias
2. Prioritized Experience Replay - Samples important transitions more
3. Dueling Networks - Separates value and advantage streams
4. Multi-step Learning - Uses n-step returns
5. Distributional RL - Models return distribution
6. Noisy Nets - Parameter noise for exploration

Author: Sandhya (NIT Sikkim)
Reference: Hessel et al., "Rainbow: Combining Improvements in Deep Reinforcement Learning"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
import random


@dataclass
class RainbowConfig:
    """Configuration for Rainbow DQN
    
    FAST CONVERGENCE: Optimized for 3-5x faster training.
    Prioritizes speed over maximum performance.
    """
    
    # Network architecture - smaller for faster training
    hidden_dim: int = 128  # FAST: Was 256
    
    # Learning - faster learning rate with aggressive decay
    lr: float = 0.0003  # FAST: Was 0.0001
    lr_decay: float = 0.9995  # FAST: Was 0.9999
    lr_min: float = 0.00005  # FAST: Was 0.00001
    gamma: float = 0.99
    batch_size: int = 256  # FAST: Was 128, larger for GPU efficiency
    
    # Replay buffer - faster warmup
    buffer_size: int = 100000  # Reduced from 200000
    min_buffer_size: int = 2000  # FAST: Was 5000, start learning earlier
    
    # Target network - can use soft or hard updates
    target_update_freq: int = 500  # FAST: Was 2000
    use_soft_update: bool = True  # FAST: NEW - Polyak averaging
    tau: float = 0.005  # FAST: NEW - soft update rate
    
    # Double DQN - KEEP (essential for stability)
    use_double_dqn: bool = True
    
    # Prioritized replay - KEEP (helps with sparse rewards)
    use_prioritized_replay: bool = True
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_end: float = 1.0
    beta_frames: int = 50000  # FAST: Was 100000
    
    # Dueling - KEEP (helps value estimation)
    use_dueling: bool = True
    
    # Multi-step - shorter for faster credit assignment
    use_multistep: bool = True
    n_step: int = 3  # FAST: Was 5
    
    # Distributional - DISABLED for speed
    use_distributional: bool = False  # FAST: Was True
    num_atoms: int = 51
    v_min: float = -2.0
    v_max: float = 3.0
    
    # Noisy nets - DISABLED, use epsilon-greedy instead
    use_noisy_nets: bool = False  # FAST: Was True
    noisy_std: float = 0.5
    
    # Hybrid exploration - DISABLED for simplicity
    use_hybrid_exploration: bool = False  # FAST: Was True
    epsilon_hybrid_start: float = 0.3
    epsilon_hybrid_end: float = 0.01
    epsilon_hybrid_decay_episodes: int = 5000
    
    # Standard epsilon-greedy - FAST DECAY
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 30000  # FAST: Was 150000
    
    # Gradient clipping
    max_grad_norm: float = 10.0


class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer for parameter-space exploration.
    
    Implements factorized Gaussian noise as described in:
    "Noisy Networks for Exploration" (Fortunato et al., 2017)
    """
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Factorized noise
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize parameters"""
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        """Sample new noise"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """Factorized Gaussian noise"""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)


class RainbowNetwork(nn.Module):
    """
    Rainbow DQN Network with all improvements.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: RainbowConfig
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.num_atoms = config.num_atoms if config.use_distributional else 1
        
        # Support for distributional RL
        if config.use_distributional:
            self.register_buffer('support', torch.linspace(config.v_min, config.v_max, config.num_atoms))
        
        # Feature extraction
        Linear = NoisyLinear if config.use_noisy_nets else nn.Linear
        
        self.feature = nn.Sequential(
            nn.Linear(state_dim, config.hidden_dim),
            nn.ReLU()
        )
        
        if config.use_dueling:
            # Value stream
            self.value_stream = nn.Sequential(
                Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                Linear(config.hidden_dim // 2, self.num_atoms)
            )
            
            # Advantage stream
            self.advantage_stream = nn.Sequential(
                Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                Linear(config.hidden_dim // 2, action_dim * self.num_atoms)
            )
        else:
            self.output = nn.Sequential(
                Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                Linear(config.hidden_dim // 2, action_dim * self.num_atoms)
            )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Returns:
            If distributional: log probabilities [batch, action, atoms]
            Else: Q-values [batch, action]
        """
        batch_size = state.shape[0]
        features = self.feature(state)
        
        if self.config.use_dueling:
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            
            value = value.view(batch_size, 1, self.num_atoms)
            advantage = advantage.view(batch_size, self.action_dim, self.num_atoms)
            
            q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            q_atoms = self.output(features)
            q_atoms = q_atoms.view(batch_size, self.action_dim, self.num_atoms)
        
        if self.config.use_distributional:
            # Return log probabilities
            log_probs = F.log_softmax(q_atoms, dim=2)
            return log_probs
        else:
            return q_atoms.squeeze(-1)
    
    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """Get Q-values from distribution"""
        if self.config.use_distributional:
            log_probs = self.forward(state)
            probs = log_probs.exp()
            q_values = (probs * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=2)
            return q_values
        else:
            return self.forward(state)
    
    def reset_noise(self):
        """Reset noise in noisy layers"""
        if self.config.use_noisy_nets:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    
    Implements proportional prioritization using a sum tree.
    """
    
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        n_step: int = 1,
        gamma: float = 0.99
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.n_step = n_step
        self.gamma = gamma
        
        # Storage
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
        # N-step buffer
        self.n_step_buffer = deque(maxlen=n_step)
        
        # Priority bounds
        self.max_priority = 1.0
    
    def _get_n_step_return(self) -> Tuple:
        """Compute n-step return from buffer"""
        reward = 0.0
        for i, transition in enumerate(self.n_step_buffer):
            reward += (self.gamma ** i) * transition[2]
        
        state = self.n_step_buffer[0][0]
        action = self.n_step_buffer[0][1]
        next_state = self.n_step_buffer[-1][3]
        done = self.n_step_buffer[-1][4]
        
        return (state, action, reward, next_state, done)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add transition to buffer"""
        transition = (state, action, reward, next_state, done)
        
        if self.n_step > 1:
            self.n_step_buffer.append(transition)
            if len(self.n_step_buffer) < self.n_step:
                return
            transition = self._get_n_step_return()
        
        if self.size < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int, beta: float) -> Tuple:
        """Sample batch with priorities"""
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(self.size, batch_size, p=probs)
        
        # Importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        batch = [self.buffer[i] for i in indices]
        
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled transitions"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        return self.size


class RainbowDQNAgent:
    """
    Rainbow DQN Agent.
    
    Combines all six improvements for state-of-the-art performance.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[RainbowConfig] = None,
        device: str = "cuda"  # FAST: Default to GPU
    ):
        # GPU availability check
        if device == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA not available, falling back to CPU")
            device = "cpu"
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or RainbowConfig()
        self.device = torch.device(device)
        
        # Networks
        self.online_net = RainbowNetwork(state_dim, action_dim, self.config).to(self.device)
        self.target_net = RainbowNetwork(state_dim, action_dim, self.config).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.config.lr)
        
        # Replay buffer
        n_step = self.config.n_step if self.config.use_multistep else 1
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.config.buffer_size,
            alpha=self.config.alpha if self.config.use_prioritized_replay else 0,
            n_step=n_step,
            gamma=self.config.gamma
        )
        
        # Exploration - including hybrid exploration
        self.epsilon = self.config.epsilon_start
        self.epsilon_hybrid = self.config.epsilon_hybrid_start if self.config.use_hybrid_exploration else 0
        self.frame = 0
        self.episode_count = 0
        
        # Learning rate scheduling
        self.current_lr = self.config.lr
        
        # Metrics
        self.losses = []
        self.q_values = []
    
    def _soft_update_target(self):
        """Polyak averaging for smoother target network updates (FAST CONVERGENCE)."""
        tau = self.config.tau
        for target_param, online_param in zip(
            self.target_net.parameters(),
            self.online_net.parameters()
        ):
            target_param.data.copy_(
                tau * online_param.data + (1.0 - tau) * target_param.data
            )
    
    def update_learning_rate(self):
        """Decay learning rate after each episode."""
        self.current_lr = max(
            self.current_lr * self.config.lr_decay,
            self.config.lr_min
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr
    
    def update_epsilon(self):
        """Update epsilon for hybrid exploration after each episode."""
        self.episode_count += 1
        if self.config.use_hybrid_exploration:
            # Linear decay of hybrid epsilon
            decay_progress = min(1.0, self.episode_count / self.config.epsilon_hybrid_decay_episodes)
            self.epsilon_hybrid = self.config.epsilon_hybrid_start + \
                (self.config.epsilon_hybrid_end - self.config.epsilon_hybrid_start) * decay_progress
    
    def select_action(
        self, 
        state: np.ndarray, 
        training: bool = True,
        action_mask: Optional[np.ndarray] = None
    ) -> int:
        """
        Select action using hybrid exploration (epsilon + noisy nets) or pure noisy nets.
        
        Args:
            state: Current state observation
            training: Whether in training mode (enables exploration)
            action_mask: Optional binary mask (1=valid, 0=invalid action)
                        Invalid actions will not be selected.
        
        Returns:
            Selected action index
        """
        # Hybrid exploration: use epsilon-greedy ON TOP of noisy nets
        if training and self.config.use_hybrid_exploration:
            if random.random() < self.epsilon_hybrid:
                if action_mask is not None:
                    valid_actions = np.where(action_mask > 0)[0]
                    if len(valid_actions) > 0:
                        return np.random.choice(valid_actions)
                return random.randint(0, self.action_dim - 1)
        
        # Standard epsilon-greedy (if noisy nets disabled)
        if training and not self.config.use_noisy_nets:
            if random.random() < self.epsilon:
                if action_mask is not None:
                    valid_actions = np.where(action_mask > 0)[0]
                    if len(valid_actions) > 0:
                        return np.random.choice(valid_actions)
                return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_net.get_q_values(state_t)
            
            # Apply action mask: set invalid actions to -infinity
            if action_mask is not None:
                mask_t = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
                q_values = q_values.masked_fill(mask_t == 0, float('-inf'))
            
            action = q_values.argmax(dim=1).item()
            
            if training:
                self.q_values.append(q_values.max().item())
        
        return action
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Optional[float]:
        """Perform one update step"""
        if len(self.replay_buffer) < self.config.min_buffer_size:
            return None
        
        self.frame += 1
        
        # Compute beta for importance sampling
        beta = self.config.beta_start + (self.config.beta_end - self.config.beta_start) * \
               min(1.0, self.frame / self.config.beta_frames)
        
        # Sample batch
        states, actions, rewards, next_states, dones, indices, weights = \
            self.replay_buffer.sample(self.config.batch_size, beta)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Compute loss
        if self.config.use_distributional:
            loss, td_errors = self._compute_distributional_loss(
                states, actions, rewards, next_states, dones, weights
            )
        else:
            loss, td_errors = self._compute_dqn_loss(
                states, actions, rewards, next_states, dones, weights
            )
        
        # Optimize
        self.optimizer.zero_grad(set_to_none=True)  # GPU OPTIMIZED: Faster than zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()
        
        # Update priorities
        if self.config.use_prioritized_replay:
            priorities = td_errors.detach().abs().cpu().numpy() + 1e-6
            self.replay_buffer.update_priorities(indices, priorities)
        
        # Update target network (soft or hard)
        if getattr(self.config, 'use_soft_update', False):
            self._soft_update_target()
        elif self.frame % self.config.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        
        # Reset noise
        if self.config.use_noisy_nets:
            self.online_net.reset_noise()
            self.target_net.reset_noise()
        
        # Decay epsilon
        if not self.config.use_noisy_nets:
            self.epsilon = max(
                self.config.epsilon_end,
                self.epsilon - (self.config.epsilon_start - self.config.epsilon_end) / self.config.epsilon_decay
            )
        
        self.losses.append(loss.item())
        return loss.item()
    
    def _compute_dqn_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute standard DQN loss"""
        # Current Q values
        q_values = self.online_net.get_q_values(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q values
        with torch.no_grad():
            if self.config.use_double_dqn:
                # Double DQN: use online net to select action, target net to evaluate
                next_actions = self.online_net.get_q_values(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_net.get_q_values(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q_values = self.target_net.get_q_values(next_states).max(dim=1)[0]
            
            # N-step return
            gamma = self.config.gamma ** (self.config.n_step if self.config.use_multistep else 1)
            target_q = rewards + gamma * next_q_values * (1 - dones)
        
        # TD errors
        td_errors = target_q - q_values
        
        # Weighted Huber loss
        loss = (weights * F.smooth_l1_loss(q_values, target_q, reduction='none')).mean()
        
        return loss, td_errors
    
    def _compute_distributional_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute distributional (C51) loss"""
        batch_size = states.shape[0]
        support = self.online_net.support
        delta_z = (self.config.v_max - self.config.v_min) / (self.config.num_atoms - 1)
        
        with torch.no_grad():
            # Get next action
            if self.config.use_double_dqn:
                next_actions = self.online_net.get_q_values(next_states).argmax(dim=1)
            else:
                next_actions = self.target_net.get_q_values(next_states).argmax(dim=1)
            
            # Get next distribution
            next_log_probs = self.target_net(next_states)
            next_probs = next_log_probs.exp()
            next_probs = next_probs[range(batch_size), next_actions]
            
            # Compute target distribution
            gamma = self.config.gamma ** (self.config.n_step if self.config.use_multistep else 1)
            tz = rewards.unsqueeze(1) + gamma * (1 - dones.unsqueeze(1)) * support.unsqueeze(0)
            tz = tz.clamp(self.config.v_min, self.config.v_max)
            
            b = (tz - self.config.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()
            
            # Fix corner case
            l[(u > 0) & (l == u)] -= 1
            u[(l < (self.config.num_atoms - 1)) & (l == u)] += 1
            
            # Distribute probability mass
            target_probs = torch.zeros_like(next_probs)
            offset = torch.linspace(0, (batch_size - 1) * self.config.num_atoms, batch_size).long().unsqueeze(1).to(self.device)
            
            target_probs.view(-1).index_add_(0, (l + offset).view(-1), (next_probs * (u.float() - b)).view(-1))
            target_probs.view(-1).index_add_(0, (u + offset).view(-1), (next_probs * (b - l.float())).view(-1))
        
        # Current distribution
        log_probs = self.online_net(states)
        log_probs = log_probs[range(batch_size), actions]
        
        # Cross-entropy loss
        loss = -(target_probs * log_probs).sum(dim=1)
        
        # TD error for prioritization (use expected Q difference)
        with torch.no_grad():
            q_current = (log_probs.exp() * support).sum(dim=1)
            q_target = (target_probs * support).sum(dim=1)
            td_errors = q_target - q_current
        
        # Weighted loss
        loss = (weights * loss).mean()
        
        return loss, td_errors
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get network parameters for federated learning"""
        return {k: v.cpu().clone() for k, v in self.online_net.state_dict().items()}
    
    def set_parameters(self, params: Dict[str, torch.Tensor]):
        """Set network parameters from federated aggregation"""
        self.online_net.load_state_dict(params)
        self.target_net.load_state_dict(params)
    
    def save(self, path: str):
        """Save agent state"""
        torch.save({
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'frame': self.frame,
            'epsilon': self.epsilon,
            'config': self.config
        }, path)
    
    def load(self, path: str):
        """Load agent state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint['online_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.frame = checkpoint['frame']
        self.epsilon = checkpoint['epsilon']
    
    def get_metrics(self) -> Dict[str, float]:
        """Get training metrics"""
        return {
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'avg_q': np.mean(self.q_values[-100:]) if self.q_values else 0,
            'epsilon': self.epsilon,
            'frame': self.frame
        }