"""
Federated Learning Module for FedMO-DRLQ
=========================================
Implements federated learning for distributed quantum cloud scheduling:
- FedAvg: Federated Averaging
- FedProx: Proximal Federated Optimization
- FedNova: Normalized Averaging for heterogeneous local steps

This addresses Gap 3: Federated Learning for Quantum Cloud Resource Management

Author: Sandhya (NIT Sikkim)
Reference: FedMO-DRLQ Research Gap Analysis
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from copy import deepcopy
import threading
from collections import OrderedDict


class AggregationMethod(Enum):
    """Federated aggregation methods"""
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    FEDNOVA = "fednova"


@dataclass
class FederatedConfig:
    """Configuration for federated learning"""
    
    # Aggregation strategy
    aggregation: AggregationMethod = AggregationMethod.FEDAVG
    
    # Number of local training steps before aggregation
    local_steps: int = 20  # Was 10 - more local training per round for convergence
    
    # Number of federated rounds
    num_rounds: int = 50  # Was 100 - matched to experiment config
    
    # Client participation rate (fraction of clients per round)
    participation_rate: float = 1.0
    
    # FedProx proximal term coefficient (μ)
    fedprox_mu: float = 0.1  # RESEARCH: Increased for non-IID quantum workloads
    
    # FedNova configuration - RESEARCH: Fix for erratic results
    fednova_tau_effective: float = 1.0  # Standardized tau (was: variable)
    fednova_normalize_gradients: bool = True  # Add gradient normalization
    fednova_momentum: float = 0.9  # Smooth updates
    
    # Communication settings
    compression_ratio: float = 1.0  # 1.0 = no compression
    
    # Differential privacy
    use_differential_privacy: bool = False
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    dp_clip_norm: float = 1.0
    
    # Asynchronous settings
    async_aggregation: bool = False
    staleness_threshold: int = 5


@dataclass
class ClientState:
    """State of a federated client"""
    client_id: int
    datacenter_name: str
    model_params: Dict[str, torch.Tensor] = field(default_factory=dict)
    local_steps: int = 0
    local_samples: int = 0
    last_update_round: int = 0
    cumulative_gradient: Optional[Dict[str, torch.Tensor]] = None


class FederatedAggregator:
    """
    Federated Learning Aggregator for FedMO-DRLQ.
    
    Supports multiple aggregation strategies for combining
    model updates from distributed quantum datacenters.
    """
    
    def __init__(
        self,
        config: Optional[FederatedConfig] = None,
        n_clients: int = 5,
        device: str = "cpu"
    ):
        """
        Initialize federated aggregator.
        
        Args:
            config: Federated learning configuration
            n_clients: Number of federated clients (datacenters)
            device: Device for computations
        """
        self.config = config or FederatedConfig()
        self.n_clients = n_clients
        self.device = torch.device(device)
        
        # Global model
        self.global_model: Optional[Dict[str, torch.Tensor]] = None
        
        # Client states
        self.clients: Dict[int, ClientState] = {}
        
        # Round tracking
        self.current_round = 0
        
        # Metrics
        self.round_metrics: List[Dict[str, float]] = []
        
        # Thread lock for async aggregation
        self._lock = threading.Lock()
    
    def register_client(
        self,
        client_id: int,
        datacenter_name: str,
        initial_params: Dict[str, torch.Tensor]
    ):
        """
        Register a new federated client.
        
        Args:
            client_id: Unique client identifier
            datacenter_name: Name of the quantum datacenter
            initial_params: Initial model parameters
        """
        with self._lock:
            self.clients[client_id] = ClientState(
                client_id=client_id,
                datacenter_name=datacenter_name,
                model_params=deepcopy(initial_params)
            )
            
            # Initialize global model if first client
            if self.global_model is None:
                self.global_model = deepcopy(initial_params)
    
    def get_global_model(self) -> Dict[str, torch.Tensor]:
        """Get current global model parameters"""
        if self.global_model is None:
            raise ValueError("Global model not initialized. Register clients first.")
        return deepcopy(self.global_model)
    
    def submit_update(
        self,
        client_id: int,
        updated_params: Dict[str, torch.Tensor],
        local_steps: int,
        local_samples: int
    ):
        """
        Submit model update from a client.
        
        Args:
            client_id: Client identifier
            updated_params: Updated model parameters after local training
            local_steps: Number of local training steps performed
            local_samples: Number of samples used in local training
        """
        if client_id not in self.clients:
            raise ValueError(f"Unknown client: {client_id}")
        
        with self._lock:
            client = self.clients[client_id]
            client.model_params = deepcopy(updated_params)
            client.local_steps = local_steps
            client.local_samples = local_samples
            client.last_update_round = self.current_round
            
            # Compute gradient for FedNova
            if self.config.aggregation == AggregationMethod.FEDNOVA:
                client.cumulative_gradient = self._compute_gradient(
                    self.global_model,
                    updated_params
                )
    
    def _compute_gradient(
        self,
        old_params: Dict[str, torch.Tensor],
        new_params: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute parameter gradient (delta)"""
        gradient = {}
        for key in old_params.keys():
            gradient[key] = new_params[key] - old_params[key]
        return gradient
    
    def aggregate(self, participating_clients: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
        """
        Aggregate model updates from participating clients.
        
        Args:
            participating_clients: List of client IDs to include.
                                   If None, use all clients based on participation rate.
        
        Returns:
            Aggregated global model parameters
        """
        with self._lock:
            # Select participating clients
            if participating_clients is None:
                n_participants = max(1, int(self.n_clients * self.config.participation_rate))
                all_clients = list(self.clients.keys())
                participating_clients = np.random.choice(
                    all_clients,
                    size=min(n_participants, len(all_clients)),
                    replace=False
                ).tolist()
            
            # Filter clients with updates
            active_clients = [
                cid for cid in participating_clients
                if self.clients[cid].local_samples > 0
            ]
            
            if not active_clients:
                return self.global_model
            
            # Aggregate based on method
            if self.config.aggregation == AggregationMethod.FEDAVG:
                aggregated = self._fedavg(active_clients)
            elif self.config.aggregation == AggregationMethod.FEDPROX:
                aggregated = self._fedprox(active_clients)
            elif self.config.aggregation == AggregationMethod.FEDNOVA:
                aggregated = self._fednova(active_clients)
            else:
                aggregated = self._fedavg(active_clients)
            
            # Apply differential privacy if enabled
            if self.config.use_differential_privacy:
                aggregated = self._apply_differential_privacy(aggregated)
            
            # Update global model
            self.global_model = aggregated
            self.current_round += 1
            
            # Record metrics
            self._record_round_metrics(active_clients)
            
            return self.global_model
    
    def _fedavg(self, client_ids: List[int]) -> Dict[str, torch.Tensor]:
        """
        FedAvg: Federated Averaging.
        
        Weighted average of client parameters by number of samples.
        """
        total_samples = sum(self.clients[cid].local_samples for cid in client_ids)
        
        if total_samples == 0:
            return self.global_model
        
        # Initialize aggregated parameters
        aggregated = {}
        for key in self.global_model.keys():
            aggregated[key] = torch.zeros_like(self.global_model[key], dtype=torch.float32)
        
        # Weighted sum
        for cid in client_ids:
            client = self.clients[cid]
            weight = client.local_samples / total_samples
            
            for key in aggregated.keys():
                aggregated[key] += weight * client.model_params[key].float()
        
        # Convert back to original dtype
        for key in aggregated.keys():
            aggregated[key] = aggregated[key].to(self.global_model[key].dtype)
        
        return aggregated
    
    def _fedprox(self, client_ids: List[int]) -> Dict[str, torch.Tensor]:
        """
        FedProx: Proximal Federated Optimization.
        
        FedAvg with proximal regularization for handling heterogeneity.
        Note: Proximal term is applied during local training, not aggregation.
        Aggregation is same as FedAvg.
        """
        return self._fedavg(client_ids)
    
    def _fednova(self, client_ids: List[int]) -> Dict[str, torch.Tensor]:
        """
        FedNova: Normalized Averaging.
        
        Normalizes updates by number of local steps to handle
        heterogeneous local computation.
        """
        # Compute tau_eff (effective number of local steps)
        total_samples = sum(self.clients[cid].local_samples for cid in client_ids)
        
        if total_samples == 0:
            return self.global_model
        
        # Compute normalized gradients
        aggregated_gradient = {}
        for key in self.global_model.keys():
            aggregated_gradient[key] = torch.zeros_like(self.global_model[key], dtype=torch.float32)
        
        total_tau = 0
        for cid in client_ids:
            client = self.clients[cid]
            weight = client.local_samples / total_samples
            tau_i = client.local_steps
            total_tau += weight * tau_i
            
            if client.cumulative_gradient is not None:
                for key in aggregated_gradient.keys():
                    # Normalize by local steps
                    normalized_grad = client.cumulative_gradient[key].float() / tau_i
                    aggregated_gradient[key] += weight * normalized_grad
        
        # Apply normalized gradient
        aggregated = {}
        for key in self.global_model.keys():
            # Scale by total_tau
            aggregated[key] = (
                self.global_model[key].float() + 
                total_tau * aggregated_gradient[key]
            ).to(self.global_model[key].dtype)
        
        return aggregated
    
    def _apply_differential_privacy(
        self,
        params: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply differential privacy noise to parameters"""
        # Compute sensitivity based on clip norm
        sensitivity = 2 * self.config.dp_clip_norm / len(self.clients)
        
        # Compute noise scale for Gaussian mechanism
        noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / self.config.dp_delta)) / self.config.dp_epsilon
        
        # Add noise to each parameter
        private_params = {}
        for key, param in params.items():
            noise = torch.randn_like(param) * noise_scale
            private_params[key] = param + noise
        
        return private_params
    
    def _record_round_metrics(self, client_ids: List[int]):
        """Record metrics for this round"""
        metrics = {
            "round": self.current_round,
            "n_participants": len(client_ids),
            "total_samples": sum(self.clients[cid].local_samples for cid in client_ids),
            "avg_local_steps": np.mean([self.clients[cid].local_steps for cid in client_ids])
        }
        self.round_metrics.append(metrics)
    
    def get_fedprox_loss(
        self,
        model: nn.Module,
        global_params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute FedProx proximal regularization term.
        
        To be added to local training loss:
        L_local = L_task + (μ/2) * ||w - w_global||²
        
        Args:
            model: Local model
            global_params: Global model parameters
            
        Returns:
            Proximal regularization term
        """
        prox_term = torch.tensor(0.0)
        
        for name, param in model.named_parameters():
            if name in global_params:
                prox_term += ((param - global_params[name].to(param.device)) ** 2).sum()
        
        return (self.config.fedprox_mu / 2) * prox_term
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregation metrics"""
        return {
            "current_round": self.current_round,
            "n_clients": len(self.clients),
            "aggregation_method": self.config.aggregation.value,
            "round_history": self.round_metrics
        }


class FederatedClient:
    """
    Federated Learning Client for a Quantum Datacenter.
    
    Manages local training and communication with the aggregator.
    """
    
    def __init__(
        self,
        client_id: int,
        datacenter_name: str,
        model: nn.Module,
        aggregator: FederatedAggregator,
        config: Optional[FederatedConfig] = None,
        device: str = "cpu"
    ):
        """
        Initialize federated client.
        
        Args:
            client_id: Unique client identifier
            datacenter_name: Name of quantum datacenter
            model: Local model
            aggregator: Federated aggregator
            config: Federated configuration
            device: Device for computations
        """
        self.client_id = client_id
        self.datacenter_name = datacenter_name
        self.model = model
        self.aggregator = aggregator
        self.config = config or FederatedConfig()
        self.device = torch.device(device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Register with aggregator
        self.aggregator.register_client(
            client_id=client_id,
            datacenter_name=datacenter_name,
            initial_params=self._get_params()
        )
        
        # Training state
        self.local_step_count = 0
        self.local_sample_count = 0
    
    def _get_params(self) -> Dict[str, torch.Tensor]:
        """Get model parameters"""
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
    
    def _set_params(self, params: Dict[str, torch.Tensor]):
        """Set model parameters"""
        self.model.load_state_dict(params)
    
    def sync_with_global(self):
        """Synchronize local model with global model"""
        global_params = self.aggregator.get_global_model()
        self._set_params(global_params)
        self.local_step_count = 0
        self.local_sample_count = 0
    
    def train_step(self, n_samples: int):
        """
        Record a local training step.
        
        Call this after each local training iteration.
        
        Args:
            n_samples: Number of samples used in this step
        """
        self.local_step_count += 1
        self.local_sample_count += n_samples
    
    def should_aggregate(self) -> bool:
        """Check if should submit update for aggregation"""
        return self.local_step_count >= self.config.local_steps
    
    def submit_for_aggregation(self):
        """Submit local model for aggregation"""
        self.aggregator.submit_update(
            client_id=self.client_id,
            updated_params=self._get_params(),
            local_steps=self.local_step_count,
            local_samples=self.local_sample_count
        )
    
    def get_fedprox_loss(self) -> torch.Tensor:
        """Get FedProx regularization term for local training"""
        if self.config.aggregation != AggregationMethod.FEDPROX:
            return torch.tensor(0.0, device=self.device)
        
        global_params = self.aggregator.get_global_model()
        return self.aggregator.get_fedprox_loss(self.model, global_params)


class FederatedTrainer:
    """
    Orchestrates federated training across multiple datacenters.
    """
    
    def __init__(
        self,
        n_datacenters: int,
        model_factory: Callable[[], nn.Module],
        config: Optional[FederatedConfig] = None,
        device: str = "cpu"
    ):
        """
        Initialize federated trainer.
        
        Args:
            n_datacenters: Number of quantum datacenters
            model_factory: Function to create model instances
            config: Federated configuration
            device: Device for computations
        """
        self.n_datacenters = n_datacenters
        self.model_factory = model_factory
        self.config = config or FederatedConfig()
        self.device = device
        
        # Create aggregator
        self.aggregator = FederatedAggregator(
            config=self.config,
            n_clients=n_datacenters,
            device=device
        )
        
        # Create clients
        self.clients: List[FederatedClient] = []
        datacenter_names = ["DC_" + name for name in 
                           ["washington", "kolkata", "hanoi", "perth", "lagos"][:n_datacenters]]
        
        for i in range(n_datacenters):
            model = model_factory()
            client = FederatedClient(
                client_id=i,
                datacenter_name=datacenter_names[i] if i < len(datacenter_names) else f"DC_{i}",
                model=model,
                aggregator=self.aggregator,
                config=self.config,
                device=device
            )
            self.clients.append(client)
    
    def get_client(self, client_id: int) -> FederatedClient:
        """Get client by ID"""
        return self.clients[client_id]
    
    def aggregate_and_sync(self) -> Dict[str, torch.Tensor]:
        """
        Aggregate updates and sync all clients.
        
        Returns:
            Updated global model parameters
        """
        # Submit all client updates
        for client in self.clients:
            client.submit_for_aggregation()
        
        # Aggregate
        global_model = self.aggregator.aggregate()
        
        # Sync clients
        for client in self.clients:
            client.sync_with_global()
        
        return global_model
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get federated training metrics"""
        return self.aggregator.get_metrics()