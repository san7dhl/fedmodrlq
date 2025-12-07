"""
FedMO-DRLQ Enhanced Environment
================================
Gymnasium environment for quantum cloud scheduling with:
- Error-aware state representation (Gap 1)
- Multi-objective reward function (Gap 2)
- Support for federated learning (Gap 3)
- Scalable design for multi-datacenter scenarios (Gap 4)

Author: Sandhya (NIT Sikkim)
"""

import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Dict
import numpy as np
from typing import Dict as TypingDict, List, Optional, Tuple, Any
import pandas as pd
from dataclasses import dataclass
import ast

# Import core modules
from ..core.nisq_error_metrics import (
    NISQDeviceMetrics, NISQErrorSimulator, FidelityEstimator, 
    EnergyEstimator, IBM_DEVICE_CONFIGS, get_error_aware_state_dim
)
from ..core.multi_objective_reward import (
    MultiObjectiveReward, MultiObjectiveConfig, ScalarizationMethod,
    create_balanced_reward
)


@dataclass
class QTask:
    """Quantum Task representation"""
    id: str
    arrival_time: float
    qubit_number: int
    circuit_depth: int
    gates: TypingDict[str, int]
    algorithm: str
    status: str = "queued"
    assigned_node: Optional[int] = None
    waiting_time: float = 0.0
    execution_time: float = 0.0
    rescheduling_count: int = 0
    estimated_fidelity: float = 0.0
    energy_consumed: float = 0.0


@dataclass
class QNode:
    """Quantum Node representation with NISQ error metrics"""
    id: int
    name: str
    num_qubits: int
    clops: float
    error_simulator: NISQErrorSimulator
    next_available_time: float = 0.0
    total_tasks_executed: int = 0
    total_energy_consumed: float = 0.0
    
    def get_error_state(self) -> np.ndarray:
        """Get error-aware state vector for this node"""
        return self.error_simulator.get_state_vector()
    
    def get_metrics(self) -> NISQDeviceMetrics:
        """Get current NISQ metrics"""
        return self.error_simulator.get_metrics()


class FedMOEnv(gym.Env):
    """
    FedMO-DRLQ Gymnasium Environment.
    
    Features:
    1. Error-Aware State Space:
       - Task features: [arrival_time, qubits, depth, rescheduling_count]
       - Per-node NISQ metrics: [ε²q, T₂, ε_r, t_cal, ε̇₂q, QV, CLOPS]
       - Per-node scheduling: [next_available_time]
    
    2. Multi-Objective Reward:
       - Completion time minimization
       - Fidelity maximization
       - Energy minimization
    
    3. Federated Learning Support:
       - Datacenter ID for multi-datacenter scenarios
       - Local/global model parameter interfaces
    """
    
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}
    
    # IBM Quantum node configurations
    DEFAULT_NODES = ["washington", "kolkata", "hanoi", "perth", "lagos"]
    
    def __init__(
        self,
        dataset_path: str,
        n_qnodes: int = 5,
        n_qtasks_per_episode: int = 25,
        node_names: Optional[List[str]] = None,
        mo_config: Optional[MultiObjectiveConfig] = None,
        include_error_features: bool = True,
        datacenter_id: int = 0,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize FedMO environment.
        
        Args:
            dataset_path: Path to quantum circuit dataset CSV
            n_qnodes: Number of quantum nodes
            n_qtasks_per_episode: Tasks per episode
            node_names: Names of IBM quantum devices to simulate
            mo_config: Multi-objective reward configuration
            include_error_features: Whether to include NISQ error features in state
            datacenter_id: ID for federated learning
            seed: Random seed
            render_mode: Rendering mode
        """
        super().__init__()
        
        # Configuration
        self.n_qnodes = n_qnodes
        self.n_qtasks_per_episode = n_qtasks_per_episode
        self.include_error_features = include_error_features
        self.datacenter_id = datacenter_id
        self.render_mode = render_mode
        
        # Node configuration
        self.node_names = node_names or self.DEFAULT_NODES[:n_qnodes]
        assert len(self.node_names) == n_qnodes
        
        # Random number generator
        self.rng = np.random.default_rng(seed)
        self._seed = seed
        
        # Load dataset
        self.dataset_path = dataset_path
        self._load_dataset()
        
        # Initialize quantum nodes
        self.qnodes: List[QNode] = []
        self._setup_qnodes()
        
        # Multi-objective reward
        self.mo_config = mo_config or MultiObjectiveConfig()
        self.reward_calculator = MultiObjectiveReward(self.mo_config)
        
        # Define observation space
        self.obs_dim = self._compute_obs_dim()
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
        )
        
        # Define action space
        self.action_space = Discrete(n_qnodes)
        
        # Episode state
        self.current_round = 0
        self.current_qtask: Optional[QTask] = None
        self.qtask_queue: List[QTask] = []
        self.completed_tasks: List[QTask] = []
        self.episode_step = 0
        
        # Metrics
        self.episode_rewards: List[float] = []
        self.episode_fidelities: List[float] = []
        self.episode_times: List[float] = []
        self.episode_energies: List[float] = []
    
    def _load_dataset(self):
        """Load quantum circuit dataset"""
        self.df = pd.read_csv(self.dataset_path)
        
        # Extract unique subsets
        self.subsets = self.df['subset'].unique()
        self.n_subsets = len(self.subsets)
    
    def _setup_qnodes(self):
        """Initialize quantum nodes with error simulators"""
        self.qnodes = []
        
        for i, name in enumerate(self.node_names):
            error_sim = NISQErrorSimulator(name, seed=self._seed)
            
            # Get base metrics
            metrics = error_sim.get_metrics()
            
            qnode = QNode(
                id=i,
                name=name,
                num_qubits=metrics.num_qubits,
                clops=metrics.clops,
                error_simulator=error_sim
            )
            self.qnodes.append(qnode)
    
    def _compute_obs_dim(self) -> int:
        """Compute observation dimension"""
        # Task features: [arrival_time, qubits, depth, rescheduling_count]
        task_dim = 4
        
        # Per-node features
        if self.include_error_features:
            # Error state (7) + scheduling state (1)
            per_node_dim = get_error_aware_state_dim() + 1
        else:
            # Just scheduling state: [qubits, clops, next_available]
            per_node_dim = 3
        
        return task_dim + self.n_qnodes * per_node_dim
    
    def _get_obs(self) -> np.ndarray:
        """Get current observation"""
        # Task features
        if self.current_qtask is None:
            task_obs = np.zeros(4, dtype=np.float32)
        else:
            task_obs = np.array([
                self.current_qtask.arrival_time / 1000.0,  # Normalize
                self.current_qtask.qubit_number / 127.0,  # Normalize to max qubits
                self.current_qtask.circuit_depth / 10000.0,  # Normalize
                self.current_qtask.rescheduling_count / 10.0  # Normalize
            ], dtype=np.float32)
        
        # Node features
        node_obs = []
        for qnode in self.qnodes:
            if self.include_error_features:
                # Error-aware state vector + next available time
                error_state = qnode.get_error_state()
                scheduling_state = np.array([qnode.next_available_time / 1000.0])
                node_obs.append(np.concatenate([error_state, scheduling_state]))
            else:
                # Basic state
                node_obs.append(np.array([
                    qnode.num_qubits / 127.0,
                    qnode.clops / 5000.0,
                    qnode.next_available_time / 1000.0
                ], dtype=np.float32))
        
        node_obs_flat = np.concatenate(node_obs).astype(np.float32)
        
        return np.concatenate([task_obs, node_obs_flat])
    
    def _generate_tasks(self):
        """Generate tasks for current episode"""
        # Select subset for this round
        subset_id = (self.current_round % self.n_subsets) + 1
        subset_df = self.df[self.df['subset'] == subset_id]
        
        # Handle empty subset - fall back to entire dataset
        if len(subset_df) == 0:
            subset_df = self.df
        
        # Handle case where entire dataset might still be empty
        if len(subset_df) == 0:
            raise ValueError(f"Dataset is empty, cannot generate tasks")
        
        # Sample tasks with replacement if needed
        n_available = len(subset_df)
        if n_available < self.n_qtasks_per_episode:
            # Not enough tasks, sample with replacement
            sampled_df = subset_df.sample(n=self.n_qtasks_per_episode, replace=True, random_state=self._seed)
        else:
            # Enough tasks, sample without replacement
            sampled_df = subset_df.sample(n=self.n_qtasks_per_episode, random_state=self._seed)
        
        # Generate arrival times
        base_time = self.current_round * 60.0
        arrival_times = self.rng.uniform(base_time + 0.1, base_time + 59.9, size=len(sampled_df))
        arrival_times.sort()
        
        # Create QTask objects
        self.qtask_queue = []
        for i, (_, row) in enumerate(sampled_df.iterrows()):
            # Find suitable target (prefer 127-qubit backend)
            circuit_depth = -1
            gates = {}
            qubits = row['original_width']
            
            # Try to find a compatible backend
            for backend in ['ibmq127', 'ibmq27', 'ibmq16', 'ibmq7']:
                depth_col = f'{backend}_depth'
                gates_col = f'{backend}_gates'
                if depth_col in row and row[depth_col] != -1:
                    circuit_depth = row[depth_col]
                    try:
                        gates = ast.literal_eval(row[gates_col]) if isinstance(row[gates_col], str) else {}
                    except:
                        gates = {}
                    break
            
            if circuit_depth == -1:
                circuit_depth = row['original_depth']
                try:
                    gates = ast.literal_eval(row['original_gates']) if isinstance(row['original_gates'], str) else {}
                except:
                    gates = {}
            
            task = QTask(
                id=f"{self.current_round:04d}{i:02d}",
                arrival_time=arrival_times[i],
                qubit_number=int(qubits),
                circuit_depth=int(circuit_depth),
                gates=gates,
                algorithm=row['algorithm']
            )
            self.qtask_queue.append(task)
        
        # Set current task
        if self.qtask_queue:
            self.current_qtask = self.qtask_queue.pop(0)
    
    def _can_execute_on_node(self, task: QTask, node: QNode) -> bool:
        """Check if task can execute on node"""
        return task.qubit_number <= node.num_qubits
    
    def _compute_execution_metrics(
        self,
        task: QTask,
        node: QNode
    ) -> Tuple[float, float, float, float]:
        """
        Compute execution metrics for task on node.
        
        Returns:
            Tuple of (waiting_time, execution_time, fidelity, energy)
        """
        # Waiting time
        waiting_time = max(0, node.next_available_time - task.arrival_time)
        
        # Execution time (based on circuit depth and CLOPS)
        n_gates = sum(task.gates.values()) if task.gates else task.circuit_depth
        execution_time = (n_gates * 1024) / node.clops  # 1024 shots default
        
        # Fidelity estimation
        metrics = node.get_metrics()
        fidelity = FidelityEstimator.estimate_from_circuit_info(
            metrics=metrics,
            circuit_gates=task.gates,
            circuit_depth=task.circuit_depth
        )
        
        # Energy estimation
        energy = EnergyEstimator.estimate_energy(
            metrics=metrics,
            n_total_gates=n_gates,
            execution_time_seconds=execution_time
        )
        
        return waiting_time, execution_time, fidelity, energy
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, TypingDict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Index of quantum node to assign task to
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        assert self.action_space.contains(action)
        assert self.current_qtask is not None
        
        node = self.qnodes[action]
        task = self.current_qtask
        
        # Check if task can execute on node
        if not self._can_execute_on_node(task, node):
            task.rescheduling_count += 1
            
            # After 3 rescheduling attempts, auto-correct to best valid node
            # This prevents infinite loops and focuses training on scheduling optimization
            if task.rescheduling_count >= 3:
                valid_nodes = [n for n in self.qnodes if self._can_execute_on_node(task, n)]
                if valid_nodes:
                    # Pick valid node with earliest availability (MCT heuristic)
                    node = min(valid_nodes, key=lambda n: n.next_available_time)
                    action = node.id
                    # Small penalty for needing auto-correction
                    auto_correct_penalty = -0.05
                else:
                    # No valid nodes at all - skip this task
                    if self.qtask_queue:
                        self.current_qtask = self.qtask_queue.pop(0)
                    else:
                        self.current_qtask = None
                    obs = self._get_obs()
                    terminated = self.current_qtask is None
                    return obs, -0.5, terminated, False, {"skipped": True, "reason": "no_valid_node"}
            else:
                # Normal rescheduling for first few attempts
                task.arrival_time += 1.0
                
                # Re-insert into queue
                idx = 0
                while idx < len(self.qtask_queue) and self.qtask_queue[idx].arrival_time < task.arrival_time:
                    idx += 1
                self.qtask_queue.insert(idx, task)
                
                # Penalty reward
                reward = -0.1
                
                # Get next task
                if self.qtask_queue:
                    self.current_qtask = self.qtask_queue.pop(0)
                else:
                    self.current_qtask = None
                
                obs = self._get_obs()
                terminated = self.current_qtask is None
                
                return obs, reward, terminated, False, {"rescheduled": True}
        else:
            auto_correct_penalty = 0.0
        
        # Compute execution metrics
        waiting_time, execution_time, fidelity, energy = self._compute_execution_metrics(task, node)
        
        # Update task
        task.assigned_node = action
        task.waiting_time = waiting_time
        task.execution_time = execution_time
        task.estimated_fidelity = fidelity
        task.energy_consumed = energy
        task.status = "completed"
        
        # Update node
        completion_time = waiting_time + execution_time
        node.next_available_time = task.arrival_time + completion_time
        node.total_tasks_executed += 1
        node.total_energy_consumed += energy
        
        # Update error simulator (simulate calibration drift)
        node.error_simulator.update(dt=completion_time)
        
        # Move to completed
        self.completed_tasks.append(task)
        
        # Compute multi-objective reward
        reward = self.reward_calculator.compute_reward(
            completion_time=completion_time,
            fidelity=fidelity,
            energy=energy
        ) + auto_correct_penalty  # Add penalty if auto-correction was needed
        
        # Store metrics
        self.episode_rewards.append(reward)
        self.episode_fidelities.append(fidelity)
        self.episode_times.append(completion_time)
        self.episode_energies.append(energy)
        
        # Get next task
        if self.qtask_queue:
            self.current_qtask = self.qtask_queue.pop(0)
        else:
            self.current_qtask = None
        
        # Update step counter
        self.episode_step += 1
        
        # Check termination
        terminated = self.current_qtask is None
        
        # Build info
        info = {
            "task_id": task.id,
            "assigned_node": action,
            "waiting_time": waiting_time,
            "execution_time": execution_time,
            "completion_time": completion_time,
            "fidelity": fidelity,
            "energy": energy,
            "rescheduling_count": task.rescheduling_count
        }
        
        obs = self._get_obs()
        
        return obs, reward, terminated, False, info
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[TypingDict[str, Any]] = None
    ) -> Tuple[np.ndarray, TypingDict[str, Any]]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            options: Reset options
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._seed = seed
        
        # Reset nodes
        for qnode in self.qnodes:
            qnode.next_available_time = 0.0
            qnode.error_simulator.calibrate()
        
        # Reset episode state
        self.current_round += 1
        self.episode_step = 0
        self.completed_tasks = []
        
        # Reset metrics
        self.episode_rewards = []
        self.episode_fidelities = []
        self.episode_times = []
        self.episode_energies = []
        
        # Reset reward calculator
        self.reward_calculator.reset_history()
        
        # Generate new tasks
        self._generate_tasks()
        
        obs = self._get_obs()
        info = {"round": self.current_round}
        
        return obs, info
    
    def get_episode_summary(self) -> TypingDict[str, float]:
        """Get summary statistics for the episode"""
        if not self.episode_rewards:
            return {}
        
        return {
            "total_reward": sum(self.episode_rewards),
            "mean_reward": np.mean(self.episode_rewards),
            "mean_fidelity": np.mean(self.episode_fidelities),
            "mean_completion_time": np.mean(self.episode_times),
            "total_energy": sum(self.episode_energies),
            "n_tasks": len(self.completed_tasks),
            "pareto_front_size": len(self.reward_calculator.get_pareto_front())
        }
    
    def get_action_mask(self) -> np.ndarray:
        """
        Get mask of valid actions for current task.
        
        Returns:
            np.ndarray: Binary mask where 1=valid action, 0=invalid action.
                       Invalid means task qubits > node capacity.
        """
        if self.current_qtask is None:
            return np.ones(self.n_qnodes, dtype=np.float32)
        
        mask = np.array([
            1.0 if self._can_execute_on_node(self.current_qtask, node) else 0.0
            for node in self.qnodes
        ], dtype=np.float32)
        
        # Ensure at least one valid action (fallback to highest-capacity node)
        if mask.sum() == 0:
            mask[0] = 1.0
        
        return mask
    
    def get_datacenter_id(self) -> int:
        """Get datacenter ID for federated learning"""
        return self.datacenter_id
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            print(self._render_ansi())
    
    def _render_ansi(self) -> str:
        """Render environment state as string"""
        lines = [
            f"=== FedMO-DRLQ Environment (Datacenter {self.datacenter_id}) ===",
            f"Round: {self.current_round}, Step: {self.episode_step}",
            f"Tasks in queue: {len(self.qtask_queue)}",
            ""
        ]
        
        if self.current_qtask:
            lines.extend([
                f"Current Task: {self.current_qtask.id}",
                f"  Qubits: {self.current_qtask.qubit_number}",
                f"  Depth: {self.current_qtask.circuit_depth}",
                f"  Algorithm: {self.current_qtask.algorithm}",
                ""
            ])
        
        lines.append("Quantum Nodes:")
        for node in self.qnodes:
            metrics = node.get_metrics()
            lines.append(
                f"  {node.name}: {node.num_qubits}q, "
                f"ε²q={metrics.two_qubit_error_median:.4f}, "
                f"T₂={metrics.t2_median:.1f}μs, "
                f"next_avail={node.next_available_time:.2f}s"
            )
        
        if self.episode_rewards:
            lines.extend([
                "",
                f"Episode Stats: mean_R={np.mean(self.episode_rewards):.3f}, "
                f"mean_F={np.mean(self.episode_fidelities):.3f}"
            ])
        
        return "\n".join(lines)
    
    def close(self):
        """Clean up environment"""
        pass


# Factory function for easy environment creation
def make_fedmo_env(
    dataset_path: str,
    config_name: str = "balanced",
    **kwargs
) -> FedMOEnv:
    """
    Create FedMO environment with named configuration.
    
    Args:
        dataset_path: Path to dataset
        config_name: One of "balanced", "time_focused", "fidelity_focused", "chebyshev"
        **kwargs: Additional arguments for FedMOEnv
        
    Returns:
        Configured FedMO environment
    """
    configs = {
        "balanced": MultiObjectiveConfig(
            weight_time=0.25,
            weight_fidelity=0.6,  # Prioritize fidelity for DRL learning
            weight_energy=0.15,
            scalarization=ScalarizationMethod.WEIGHTED_SUM
        ),
        "time_focused": MultiObjectiveConfig(
            weight_time=0.5,
            weight_fidelity=0.35,
            weight_energy=0.15,
            scalarization=ScalarizationMethod.WEIGHTED_SUM
        ),
        "fidelity_focused": MultiObjectiveConfig(
            weight_time=0.15,
            weight_fidelity=0.75,  # Maximum fidelity focus
            weight_energy=0.1,
            scalarization=ScalarizationMethod.WEIGHTED_SUM
        ),
        "chebyshev": MultiObjectiveConfig(
            weight_time=0.25,
            weight_fidelity=0.6,
            weight_energy=0.15,
            scalarization=ScalarizationMethod.CHEBYSHEV
        ),
        "constraint": MultiObjectiveConfig(
            weight_time=0.6,
            weight_fidelity=0.25,
            weight_energy=0.15,
            scalarization=ScalarizationMethod.CONSTRAINT_BASED,
            min_fidelity_threshold=0.8
        )
    }
    
    mo_config = configs.get(config_name, configs["balanced"])
    
    return FedMOEnv(
        dataset_path=dataset_path,
        mo_config=mo_config,
        **kwargs
    )