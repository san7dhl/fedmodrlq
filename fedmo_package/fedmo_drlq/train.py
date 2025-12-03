"""
FedMO-DRLQ Training Script
==========================
Main entry point for training FedMO-DRLQ agents.

Supports:
- Single-agent training (Rainbow DQN / PPO)
- Federated training across multiple datacenters
- Multi-objective optimization with different scalarization strategies
- Comprehensive evaluation against baselines

Usage:
    python train.py --mode single --algorithm rainbow --epochs 100
    python train.py --mode federated --n_clients 5 --rounds 50
    python train.py --mode evaluate --checkpoint path/to/model.pt

Author: Sandhya (NIT Sikkim)
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np

import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedmo_drlq.core import (
    FedMODRLQConfig,
    MultiObjectiveConfig,
    FederatedConfig,
    TrainingConfig,
    ScalarizationMethod,
    FederatedAggregation,
    get_config
)

from fedmo_drlq.envs import (
    FedMODRLQEnv,
    create_fedmo_env
)

from fedmo_drlq.agents import (
    RainbowDQNAgent,
    PPOAgent
)

from fedmo_drlq.federated import (
    FederatedServer,
    FederatedClient,
    FederatedTrainer
)

from fedmo_drlq.evaluation import (
    Evaluator,
    get_all_baselines
)


class TrainingLogger:
    """Simple logging utility"""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.log_file = os.path.join(log_dir, "training.log")
        self.metrics_file = os.path.join(log_dir, "metrics.json")
        
        self.metrics_history = []
    
    def log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, "a") as f:
            f.write(log_message + "\n")
    
    def log_metrics(self, step: int, metrics: Dict[str, float]):
        metrics_with_step = {"step": step, **metrics}
        self.metrics_history.append(metrics_with_step)
        
        # Save to file
        with open(self.metrics_file, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def save_config(self, config: Any):
        config_file = os.path.join(self.log_dir, "config.json")
        
        # Convert config to dict
        if hasattr(config, '__dict__'):
            config_dict = self._config_to_dict(config)
        else:
            config_dict = config
        
        with open(config_file, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    def _config_to_dict(self, obj) -> Dict:
        if hasattr(obj, '__dict__'):
            result = {}
            for key, value in obj.__dict__.items():
                if hasattr(value, '__dict__'):
                    result[key] = self._config_to_dict(value)
                elif hasattr(value, 'value'):  # Enum
                    result[key] = value.value
                else:
                    result[key] = value
            return result
        return obj


def create_agent(agent_type: str, state_dim: int, action_dim: int, 
                 config: TrainingConfig, device: str = "cpu"):
    """Factory function to create DRL agents"""
    if agent_type == "rainbow" or agent_type == "rainbow_dqn":
        return RainbowDQNAgent(state_dim, action_dim, config, device)
    elif agent_type == "ppo":
        return PPOAgent(state_dim, action_dim, config, device)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def train_single_agent(
    config: FedMODRLQConfig,
    logger: TrainingLogger,
    checkpoint_dir: str
) -> Dict[str, Any]:
    """
    Train a single FedMO-DRLQ agent.
    
    Args:
        config: Training configuration
        logger: Training logger
        checkpoint_dir: Directory for saving checkpoints
    
    Returns:
        Training results
    """
    logger.log("Starting single-agent training")
    logger.log(f"Algorithm: {config.training.algorithm}")
    logger.log(f"Total timesteps: {config.training.total_timesteps}")
    
    # Create environment
    env = create_fedmo_env(config=config, normalize=True)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    logger.log(f"State dimension: {state_dim}")
    logger.log(f"Action dimension: {action_dim}")
    
    # Create agent
    agent = create_agent(
        config.training.algorithm,
        state_dim,
        action_dim,
        config.training,
        config.device
    )
    
    # Training loop
    total_timesteps = config.training.total_timesteps
    eval_frequency = config.training.eval_frequency
    checkpoint_frequency = config.training.checkpoint_frequency
    
    episode = 0
    total_steps = 0
    best_reward = float('-inf')
    
    training_start = time.time()
    
    while total_steps < total_timesteps:
        obs, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        while not done and total_steps < total_timesteps:
            # Select action
            if config.training.algorithm == "ppo":
                action, log_prob, value = agent.select_action(obs)
            else:
                action = agent.select_action(obs, training=True)
                log_prob, value = None, None
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            if config.training.algorithm == "ppo":
                agent.store_transition(obs, action, reward, value, log_prob, done)
            else:
                agent.store_transition(obs, action, reward, next_obs, done)
            
            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            # Update agent
            if config.training.algorithm != "ppo":
                loss = agent.update()
            
            # Evaluation
            if total_steps % eval_frequency == 0:
                metrics = agent.get_metrics()
                metrics['episode_reward'] = episode_reward
                metrics['total_steps'] = total_steps
                
                logger.log(f"Step {total_steps}: reward={episode_reward:.4f}, "
                          f"loss={metrics.get('avg_loss', metrics.get('avg_policy_loss', 0)):.6f}")
                logger.log_metrics(total_steps, metrics)
            
            # Checkpoint
            if total_steps % checkpoint_frequency == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{total_steps}.pt")
                agent.save(checkpoint_path)
                logger.log(f"Saved checkpoint to {checkpoint_path}")
        
        # PPO update at end of episode
        if config.training.algorithm == "ppo" and len(agent.buffer) >= config.training.batch_size:
            # Get last value for bootstrapping
            with torch.no_grad():
                _, _, last_value = agent.network.get_action(
                    torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
                )
            
            update_metrics = agent.update(last_value.item())
            
            if update_metrics:
                logger.log(f"PPO Update: policy_loss={update_metrics['policy_loss']:.6f}, "
                          f"value_loss={update_metrics['value_loss']:.6f}")
        
        # Track best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            agent.save(best_path)
        
        episode += 1
    
    training_time = time.time() - training_start
    
    # Final save
    final_path = os.path.join(checkpoint_dir, "final_model.pt")
    agent.save(final_path)
    
    logger.log(f"Training completed in {training_time:.2f} seconds")
    logger.log(f"Total episodes: {episode}")
    logger.log(f"Best reward: {best_reward:.4f}")
    
    env.close()
    
    return {
        "total_episodes": episode,
        "total_timesteps": total_steps,
        "best_reward": best_reward,
        "training_time": training_time,
        "final_model_path": final_path
    }


def train_federated(
    config: FedMODRLQConfig,
    n_clients: int,
    logger: TrainingLogger,
    checkpoint_dir: str
) -> Dict[str, Any]:
    """
    Train FedMO-DRLQ with federated learning.
    
    Args:
        config: Training configuration
        n_clients: Number of federated clients (datacenters)
        logger: Training logger
        checkpoint_dir: Directory for checkpoints
    
    Returns:
        Training results
    """
    logger.log(f"Starting federated training with {n_clients} clients")
    logger.log(f"Aggregation: {config.federated.aggregation.value}")
    logger.log(f"Rounds: {config.federated.num_rounds}")
    
    # Create environments for each client
    client_envs = []
    for i in range(n_clients):
        env = create_fedmo_env(config=config, normalize=True, agent_id=i)
        client_envs.append(env)
    
    state_dim = client_envs[0].observation_space.shape[0]
    action_dim = client_envs[0].action_space.n
    
    # Create global model
    global_agent = create_agent(
        config.training.algorithm,
        state_dim,
        action_dim,
        config.training,
        config.device
    )
    
    # Initialize server
    if hasattr(global_agent, 'online_network'):
        global_model = global_agent.online_network
    else:
        global_model = global_agent.network
    
    server = FederatedServer(global_model, config.federated)
    
    # Create clients
    clients = []
    for i in range(n_clients):
        local_agent = create_agent(
            config.training.algorithm,
            state_dim,
            action_dim,
            config.training,
            config.device
        )
        
        if hasattr(local_agent, 'online_network'):
            local_model = local_agent.online_network
            optimizer = local_agent.optimizer
        else:
            local_model = local_agent.network
            optimizer = local_agent.optimizer
        
        client = FederatedClient(i, local_model, optimizer, config.federated)
        clients.append((client, local_agent, client_envs[i]))
        server.register_client(i)
    
    # Federated training loop
    training_start = time.time()
    round_results = []
    
    for round_idx in range(config.federated.num_rounds):
        round_start = time.time()
        
        # Get global parameters
        global_params = server.get_global_parameters()
        
        # Local training for each client
        client_metrics = []
        
        for client_id, (client, local_agent, env) in enumerate(clients):
            # Set global parameters
            client.set_global_parameters(global_params)
            local_agent.set_parameters(global_params)
            
            # Local training
            local_reward = 0
            local_steps = 0
            
            for _ in range(config.federated.local_steps):
                obs, _ = env.reset()
                episode_done = False
                
                while not episode_done:
                    if config.training.algorithm == "ppo":
                        action, log_prob, value = local_agent.select_action(obs)
                    else:
                        action = local_agent.select_action(obs, training=True)
                    
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    episode_done = terminated or truncated
                    
                    if config.training.algorithm == "ppo":
                        local_agent.store_transition(obs, action, reward, value, log_prob, episode_done)
                    else:
                        local_agent.store_transition(obs, action, reward, next_obs, episode_done)
                    
                    obs = next_obs
                    local_reward += reward
                    local_steps += 1
                    
                    # Update with proximal term
                    if config.training.algorithm != "ppo":
                        loss = local_agent.update()
                        if loss is not None:
                            # Add proximal term (handled in FedProx aggregator)
                            pass
                
                # PPO update
                if config.training.algorithm == "ppo" and len(local_agent.buffer) >= config.training.batch_size:
                    with torch.no_grad():
                        _, _, last_value = local_agent.network.get_action(
                            torch.FloatTensor(obs).unsqueeze(0).to(local_agent.device)
                        )
                    local_agent.update(last_value.item())
            
            # Update client tracking
            client.local_steps = local_steps
            client.total_samples = local_steps
            client.update_metrics({
                'local_reward': local_reward,
                'local_steps': local_steps
            })
            
            # Get update from agent
            if hasattr(local_agent, 'online_network'):
                client.local_model.load_state_dict(local_agent.online_network.state_dict())
            else:
                client.local_model.load_state_dict(local_agent.network.state_dict())
            
            update = client.get_update(round_idx)
            server.receive_update(update)
            
            client_metrics.append({
                'client_id': client_id,
                'reward': local_reward,
                'steps': local_steps
            })
        
        # Aggregate
        round_summary = server.aggregate_round()
        
        # Update global agent
        new_global_params = server.get_global_parameters()
        global_agent.set_parameters(new_global_params)
        
        round_time = time.time() - round_start
        
        # Log round results
        avg_reward = np.mean([m['reward'] for m in client_metrics])
        logger.log(f"Round {round_idx + 1}/{config.federated.num_rounds}: "
                  f"avg_reward={avg_reward:.4f}, time={round_time:.2f}s")
        
        logger.log_metrics(round_idx, {
            'round': round_idx,
            'avg_reward': avg_reward,
            'round_time': round_time,
            'num_clients': len(client_metrics)
        })
        
        round_results.append({
            'round': round_idx,
            'avg_reward': avg_reward,
            'client_metrics': client_metrics,
            'round_time': round_time
        })
        
        # Checkpoint
        if (round_idx + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"federated_round_{round_idx + 1}.pt")
            global_agent.save(checkpoint_path)
    
    training_time = time.time() - training_start
    
    # Final save
    final_path = os.path.join(checkpoint_dir, "federated_final.pt")
    global_agent.save(final_path)
    
    # Clean up
    for env in client_envs:
        env.close()
    
    logger.log(f"Federated training completed in {training_time:.2f} seconds")
    
    return {
        "total_rounds": config.federated.num_rounds,
        "n_clients": n_clients,
        "final_avg_reward": round_results[-1]['avg_reward'] if round_results else 0,
        "training_time": training_time,
        "final_model_path": final_path,
        "round_results": round_results
    }


def evaluate_all(
    config: FedMODRLQConfig,
    model_path: str,
    logger: TrainingLogger,
    results_dir: str
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of trained model against baselines.
    
    Args:
        config: Configuration
        model_path: Path to trained model
        logger: Logger
        results_dir: Directory for results
    
    Returns:
        Evaluation results
    """
    logger.log("Starting comprehensive evaluation")
    
    # Environment factory
    def env_factory():
        return create_fedmo_env(config=config, normalize=True)
    
    env = env_factory()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()
    
    # Load trained agent
    agent = create_agent(
        config.training.algorithm,
        state_dim,
        action_dim,
        config.training,
        config.device
    )
    agent.load(model_path)
    
    # Create evaluator
    evaluator = Evaluator(env_factory, n_eval_episodes=20)
    
    # Evaluate trained agent
    logger.log("Evaluating trained agent...")
    agent_results = evaluator.evaluate_agent(agent, "FedMO-DRLQ")
    logger.log(f"FedMO-DRLQ: reward={agent_results['avg_reward']:.4f}, "
              f"fidelity={agent_results['avg_fidelity']:.4f}")
    
    # Evaluate baselines
    baselines = get_all_baselines(n_qnodes=5, qnode_feature_dim=10)
    
    for baseline in baselines:
        logger.log(f"Evaluating {baseline.name()}...")
        baseline_results = evaluator.evaluate_baseline(baseline)
        logger.log(f"{baseline.name()}: reward={baseline_results['avg_reward']:.4f}, "
                  f"fidelity={baseline_results['avg_fidelity']:.4f}")
    
    # Generate report
    report = evaluator.generate_report()
    logger.log("\n" + report)
    
    # Save results
    results_path = os.path.join(results_dir, "evaluation_results.json")
    evaluator.save_results(results_path)
    
    report_path = os.path.join(results_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    
    # Get improvements
    improvements = evaluator.get_improvement_over_baseline("FedMO-DRLQ", "FCFS")
    
    return {
        "agent_results": agent_results,
        "improvements_over_fcfs": improvements,
        "report_path": report_path,
        "results_path": results_path
    }


def main():
    parser = argparse.ArgumentParser(description="FedMO-DRLQ Training")
    
    parser.add_argument("--mode", type=str, default="single",
                       choices=["single", "federated", "evaluate"],
                       help="Training mode")
    parser.add_argument("--algorithm", type=str, default="rainbow",
                       choices=["rainbow", "ppo"],
                       help="DRL algorithm")
    parser.add_argument("--config", type=str, default="default",
                       help="Configuration preset")
    parser.add_argument("--timesteps", type=int, default=100000,
                       help="Total training timesteps")
    parser.add_argument("--n_clients", type=int, default=5,
                       help="Number of federated clients")
    parser.add_argument("--rounds", type=int, default=50,
                       help="Number of federated rounds")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint for evaluation")
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="Output directory")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device (cpu or cuda)")
    parser.add_argument("--scalarization", type=str, default="weighted_sum",
                       choices=["weighted_sum", "chebyshev", "constraint_based"],
                       help="Multi-objective scalarization method")
    parser.add_argument("--time_weight", type=float, default=0.4,
                       help="Weight for completion time objective")
    parser.add_argument("--fidelity_weight", type=float, default=0.4,
                       help="Weight for fidelity objective")
    parser.add_argument("--energy_weight", type=float, default=0.2,
                       help="Weight for energy objective")
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.mode}_{timestamp}")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize logger
    logger = TrainingLogger(output_dir)
    
    # Load and modify configuration
    config = get_config(args.config)
    
    # Update config from args
    config.training.algorithm = args.algorithm
    config.training.total_timesteps = args.timesteps
    config.device = args.device
    
    # Set multi-objective weights
    config.multi_objective.weight_time = args.time_weight
    config.multi_objective.weight_fidelity = args.fidelity_weight
    config.multi_objective.weight_energy = args.energy_weight
    
    # Set scalarization
    scalarization_map = {
        "weighted_sum": ScalarizationMethod.WEIGHTED_SUM,
        "chebyshev": ScalarizationMethod.CHEBYSHEV,
        "constraint_based": ScalarizationMethod.CONSTRAINT_BASED
    }
    config.multi_objective.scalarization = scalarization_map[args.scalarization]
    
    # Update federated config
    config.federated.num_rounds = args.rounds
    
    # Save configuration
    logger.save_config(config)
    
    logger.log("=" * 60)
    logger.log("FedMO-DRLQ: Federated Multi-Objective Deep Reinforcement")
    logger.log("         Learning for Quantum Cloud Computing")
    logger.log("=" * 60)
    logger.log(f"Mode: {args.mode}")
    logger.log(f"Algorithm: {args.algorithm}")
    logger.log(f"Scalarization: {args.scalarization}")
    logger.log(f"Weights: time={args.time_weight}, fidelity={args.fidelity_weight}, energy={args.energy_weight}")
    logger.log(f"Output: {output_dir}")
    logger.log("=" * 60)
    
    # Run training/evaluation
    if args.mode == "single":
        results = train_single_agent(config, logger, checkpoint_dir)
        
    elif args.mode == "federated":
        results = train_federated(config, args.n_clients, logger, checkpoint_dir)
        
    elif args.mode == "evaluate":
        if args.checkpoint is None:
            logger.log("Error: --checkpoint required for evaluation mode")
            return
        
        results = evaluate_all(config, args.checkpoint, logger, output_dir)
    
    # Save final results
    results_path = os.path.join(output_dir, "final_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.log(f"\nResults saved to {results_path}")
    logger.log("Training completed successfully!")


if __name__ == "__main__":
    main()
