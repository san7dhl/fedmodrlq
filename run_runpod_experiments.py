#!/usr/bin/env python3
"""
FedMO-DRLQ Experiments - Optimized for RunPod
============================================
Includes checkpointing, GPU optimization, and progress tracking.

Usage:
    python run_runpod_experiments.py              # Run all phases
    python run_runpod_experiments.py --test       # Quick test only
    python run_runpod_experiments.py --main       # Main experiments only
    python run_runpod_experiments.py --resume     # Resume from checkpoint

Author: Sandhya (NIT Sikkim)
"""

import os
import sys
import glob
import json
import time
import argparse
import torch
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fedmo_package'))

# === CONFIGURATION ===
CHECKPOINT_FILE = "runpod_checkpoint.json"
OUTPUT_BASE = "outputs"


@dataclass
class PhaseConfig:
    """Configuration for an experiment phase"""
    name: str
    pattern: str
    n_episodes: int
    n_eval_episodes: int
    federated_rounds: int
    local_steps: int
    description: str


# Define experiment phases
PHASES = {
    "test": PhaseConfig(
        name="test",
        pattern="qsimpy/qdataset/qsimpyds_100_sub_12.csv",
        n_episodes=50,
        n_eval_episodes=10,
        federated_rounds=5,
        local_steps=3,
        description="Quick validation (~30 min)"
    ),
    "main": PhaseConfig(
        name="main",
        pattern="qsimpy/qdataset/qsimpyds_1000_sub_*.csv",
        n_episodes=50000,  # Q1 JOURNAL: Increased from 10k for proper convergence
        n_eval_episodes=50,
        federated_rounds=50,  # Q1 JOURNAL: Increased from 20
        local_steps=20,  # Q1 JOURNAL: Increased from 10
        description="Main experiments - 1000 circuits (~24-36 hrs)"
    ),
    "scalability": PhaseConfig(
        name="scalability",
        pattern="qsimpy/qdataset/qsimpyds_1900_sub_*.csv",
        n_episodes=50000,  # Q1 JOURNAL: Increased from 10k for proper convergence
        n_eval_episodes=50,
        federated_rounds=50,  # Q1 JOURNAL: Increased from 20
        local_steps=20,  # Q1 JOURNAL: Increased from 10
        description="Scalability test - 1900 circuits (~8-12 hrs)"
    ),
    "small": PhaseConfig(
        name="small",
        pattern="qsimpy/qdataset/qsimpyds_100_sub_*.csv",
        n_episodes=50000,  # Q1 JOURNAL: Increased from 10k for proper convergence
        n_eval_episodes=20,
        federated_rounds=50,  # Q1 JOURNAL: Increased from 10
        local_steps=20,  # Q1 JOURNAL: Increased from 5
        description="All small datasets (~8-12 hrs)"
    )
}


def print_banner():
    """Print startup banner"""
    print("=" * 60)
    print("  FedMO-DRLQ: Federated Multi-Objective Deep RL")
    print("  for Quantum Cloud Computing")
    print("  RunPod Experiment Runner")
    print("=" * 60)


def print_gpu_info():
    """Print GPU information"""
    print("\n[System Info]")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name}")
        print(f"  VRAM: {gpu_mem:.1f} GB")
        print(f"  CUDA: {torch.version.cuda}")
    else:
        print("  Warning: No GPU detected - running on CPU")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  NumPy: {np.__version__}")


def save_checkpoint(phase: str, dataset_idx: int, completed: List[Dict]):
    """Save progress checkpoint for resumption"""
    checkpoint = {
        "phase": phase,
        "dataset_idx": dataset_idx,
        "completed_datasets": completed,
        "timestamp": datetime.now().isoformat()
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def load_checkpoint() -> Optional[Dict]:
    """Load checkpoint if exists"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return None


def get_datasets(pattern: str) -> List[str]:
    """Get list of dataset files matching pattern"""
    if '*' in pattern:
        datasets = sorted(glob.glob(pattern))
    else:
        datasets = [pattern] if os.path.exists(pattern) else []
    return datasets


def run_single_experiment(dataset_path: str, phase: PhaseConfig, seed: int = 42) -> str:
    """Run experiment on a single dataset"""
    from experiments.run_all_experiments import ExperimentConfig, run_experiments
    
    config = ExperimentConfig(
        dataset_path=dataset_path,
        n_qnodes=5,
        n_qtasks_per_episode=25,
        n_episodes=phase.n_episodes,
        n_eval_episodes=phase.n_eval_episodes,
        n_datacenters=3,
        federated_rounds=phase.federated_rounds,
        local_steps=phase.local_steps,
        output_dir=OUTPUT_BASE,
        seed=seed
    )
    
    output_dir = run_experiments(config)
    return output_dir


def run_phase(phase_name: str, resume_from: int = 0, seed: int = 42) -> List[Dict]:
    """Run a single phase of experiments"""
    if phase_name not in PHASES:
        print(f"Unknown phase: {phase_name}")
        print(f"Available phases: {list(PHASES.keys())}")
        return []
    
    phase = PHASES[phase_name]
    
    print(f"\n{'='*60}")
    print(f"PHASE: {phase_name.upper()}")
    print(f"Description: {phase.description}")
    print(f"{'='*60}")
    
    datasets = get_datasets(phase.pattern)
    
    if not datasets:
        print(f"Warning: No datasets found for pattern: {phase.pattern}")
        return []
    
    print(f"Found {len(datasets)} dataset(s)")
    print(f"Episodes: {phase.n_episodes}, Fed Rounds: {phase.federated_rounds}")
    
    completed = []
    
    for i, dataset_path in enumerate(datasets):
        if i < resume_from:
            print(f"  [{i+1}/{len(datasets)}] Skipping (already completed): {os.path.basename(dataset_path)}")
            continue
        
        dataset_name = os.path.basename(dataset_path)
        print(f"\n[{i+1}/{len(datasets)}] Processing: {dataset_name}")
        print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
        
        start_time = time.time()
        
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            output_dir = run_single_experiment(dataset_path, phase, seed)
            elapsed = time.time() - start_time
            
            result = {
                "dataset": dataset_path,
                "dataset_name": dataset_name,
                "output_dir": output_dir,
                "elapsed_seconds": elapsed,
                "elapsed_minutes": elapsed / 60,
                "status": "success"
            }
            completed.append(result)
            
            print(f"  Completed in {elapsed/60:.1f} minutes")
            print(f"  Output: {output_dir}")
            
            save_checkpoint(phase_name, i + 1, completed)
            
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            save_checkpoint(phase_name, i, completed)
            raise
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            result = {
                "dataset": dataset_path,
                "dataset_name": dataset_name,
                "error": str(e),
                "status": "failed"
            }
            completed.append(result)
            save_checkpoint(phase_name, i + 1, completed)
            continue
    
    return completed


def generate_summary(all_results: Dict[str, List[Dict]], total_time: float):
    """Generate and save experiment summary"""
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    
    total_datasets = 0
    total_successful = 0
    
    for phase_name, results in all_results.items():
        if not results:
            continue
            
        successful = [r for r in results if r.get('status') == 'success']
        failed = [r for r in results if r.get('status') == 'failed']
        phase_time = sum(r.get('elapsed_seconds', 0) for r in successful)
        
        print(f"\n{phase_name.upper()}:")
        print(f"  Datasets: {len(successful)}/{len(results)} successful")
        print(f"  Time: {phase_time/60:.1f} minutes")
        
        if failed:
            print(f"  Failed: {[r['dataset_name'] for r in failed]}")
        
        total_datasets += len(results)
        total_successful += len(successful)
    
    print(f"\n{'='*60}")
    print(f"TOTAL: {total_successful}/{total_datasets} datasets completed")
    print(f"TOTAL TIME: {total_time/3600:.2f} hours")
    
    estimated_cost = (total_time / 3600) * 0.22
    print(f"ESTIMATED COST: ${estimated_cost:.2f} (RTX 3090 @ $0.22/hr)")
    
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    summary_path = os.path.join(OUTPUT_BASE, "experiment_summary.json")
    
    summary = {
        "total_time_hours": total_time / 3600,
        "total_time_minutes": total_time / 60,
        "estimated_cost_usd": estimated_cost,
        "total_datasets": total_datasets,
        "successful_datasets": total_successful,
        "phases": {k: v for k, v in all_results.items() if v},
        "completed_at": datetime.now().isoformat()
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="FedMO-DRLQ RunPod Experiment Runner")
    parser.add_argument('--test', action='store_true', help='Run test phase only')
    parser.add_argument('--main', action='store_true', help='Run main experiments only')
    parser.add_argument('--scalability', action='store_true', help='Run scalability tests only')
    parser.add_argument('--small', action='store_true', help='Run all small datasets')
    parser.add_argument('--all', action='store_true', help='Run all phases (default)')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--clean', action='store_true', help='Clear checkpoint and start fresh')
    
    args = parser.parse_args()
    
    print_banner()
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_gpu_info()
    
    if args.clean and os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("\nCheckpoint cleared")
    
    # Determine phases to run
    if args.test:
        phases_to_run = ["test"]
    elif args.main:
        phases_to_run = ["main"]
    elif args.scalability:
        phases_to_run = ["scalability"]
    elif args.small:
        phases_to_run = ["small"]
    else:
        phases_to_run = ["test", "main", "scalability"]
    
    print(f"\nPhases to run: {phases_to_run}")
    
    # Check for checkpoint
    checkpoint = load_checkpoint()
    start_phase_idx = 0
    resume_dataset_idx = 0
    all_results = {}
    
    if checkpoint and args.resume:
        print(f"\nFound checkpoint from {checkpoint['timestamp']}")
        print(f"  Phase: {checkpoint['phase']}, Dataset: {checkpoint['dataset_idx']}")
        
        if checkpoint['phase'] in phases_to_run:
            start_phase_idx = phases_to_run.index(checkpoint['phase'])
            resume_dataset_idx = checkpoint['dataset_idx']
            all_results[checkpoint['phase']] = checkpoint['completed_datasets']
            print("Resuming from checkpoint...")
    
    # Run experiments
    total_start = time.time()
    
    try:
        for phase_idx, phase_name in enumerate(phases_to_run):
            if phase_idx < start_phase_idx:
                continue
            
            resume_from = resume_dataset_idx if phase_idx == start_phase_idx else 0
            results = run_phase(phase_name, resume_from=resume_from, seed=args.seed)
            all_results[phase_name] = results
            
    except KeyboardInterrupt:
        print("\n\nExperiments interrupted. Progress saved to checkpoint.")
    
    total_time = time.time() - total_start
    generate_summary(all_results, total_time)
    
    print("\n" + "=" * 60)
    print("Don't forget to download results before stopping the pod!")
    print("  zip -r results.zip outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
