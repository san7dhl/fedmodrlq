# experiments/generate_paper_results.py
# Generate Publication-Ready Results for FedMO-DRLQ
# Aggregates results across multiple dataset runs.

import os
import glob
import json
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stats
from typing import List, Tuple

# Configure matplotlib for publication quality
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['grid.alpha'] = 0.3

OUTPUT_DIR = "outputs/paper"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_all_run_dirs(base_dir="outputs"):
    """
    Find all run directories from the most recent batch of experiments.
    Assumes directories are named 'run_datasetname_timestamp'.
    """
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Output directory {base_dir} not found.")
        
    # Get all run directories
    all_runs = sorted(glob.glob(os.path.join(base_dir, "run_*")))
    
    if not all_runs:
        raise FileNotFoundError("No experiment runs found in 'outputs/'")
        
    # Filter for runs created "today" to avoid mixing old/new experiments
    # (Optional: you can remove this filter if you want to aggregate everything)
    today_str = datetime.now().strftime("%Y%m%d")
    recent_runs = [r for r in all_runs if today_str in r]
    
    if not recent_runs:
        print("Warning: No runs found for today. Using all available runs.")
        recent_runs = all_runs

    print(f"Found {len(recent_runs)} run directories to aggregate.")
    return recent_runs

def aggregate_csv_results(run_dirs, filename):
    """
    Load and concatenate CSV files from multiple runs.
    Returns a single DataFrame containing all data.
    """
    all_dfs = []
    for run_dir in run_dirs:
        filepath = os.path.join(run_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['run_id'] = os.path.basename(run_dir) # Track source run
            all_dfs.append(df)
            
    if not all_dfs:
        raise FileNotFoundError(f"Could not find {filename} in any run directories.")
        
    return pd.concat(all_dfs, ignore_index=True)

# =========================================================
# TABLE 1: Baseline Comparison
# =========================================================
def generate_baseline_comparison_table(run_dirs):
    print("\n" + "=" * 60)
    print("TABLE 1: Baseline Comparison")
    print("=" * 60)
    
    # Load heuristics results
    df_heuristics = aggregate_csv_results(run_dirs, "heuristics_results.csv")
    
    # Load DRL results (Rainbow DQN) - Get final performance (last 10 episodes)
    df_drl = aggregate_csv_results(run_dirs, "rainbow_dqn_results.csv")
    # Filter for the last 10 episodes of each run to get "converged" performance
    max_ep = df_drl['episode'].max()
    df_drl_final = df_drl[df_drl['episode'] >= max_ep - 10]
    
    # Group and aggregate
    baseline_stats = df_heuristics.groupby('policy').agg({
        'mean_completion_time': ['mean', 'std'],
        'mean_fidelity': ['mean', 'std'],
        'total_energy': ['mean', 'std']
    }).reset_index()
    
    drl_stats = pd.DataFrame({
        'policy': ['FedMO-DRLQ (Ours)'],
        'mean_completion_time': [df_drl_final['mean_completion_time'].mean()],
        'mean_fidelity': [df_drl_final['mean_fidelity'].mean()],
        'total_energy': [df_drl_final['total_energy'].mean()]
    })
    
    # Format for LaTeX
    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Performance Comparison (Aggregated across " + str(len(run_dirs)) + r" datasets)}")
    latex.append(r"\label{tab:baseline_comparison}")
    latex.append(r"\begin{tabular}{lccc}")
    latex.append(r"\toprule")
    latex.append(r"\textbf{Algorithm} & \textbf{Time (s)} & \textbf{Fidelity} & \textbf{Energy (J)} \\")
    latex.append(r"\midrule")
    
    # Rows
    for _, row in baseline_stats.iterrows():
        name = row['policy'][0] if isinstance(row['policy'], tuple) else row['policy']
        time = f"{row['mean_completion_time']['mean']:.1f}"
        fid = f"{row['mean_fidelity']['mean']:.4f}"
        eng = f"{row['total_energy']['mean']:.1f}"
        latex.append(f"{name} & {time} & {fid} & {eng} \\\\")
        
    # Add DRL row
    d_time = f"\\textbf{{{drl_stats['mean_completion_time'][0]:.1f}}}"
    d_fid = f"\\textbf{{{drl_stats['mean_fidelity'][0]:.4f}}}"
    d_eng = f"\\textbf{{{drl_stats['total_energy'][0]:.1f}}}"
    latex.append(f"\\textbf{{FedMO-DRLQ}} & {d_time} & {d_fid} & {d_eng} \\\\")
    
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    
    print("\n".join(latex))
    return "\n".join(latex)

# =========================================================
# TABLE 2: Ablation Study
# =========================================================
def generate_ablation_table(run_dirs):
    print("\n" + "=" * 60)
    print("TABLE 2: Ablation Study")
    print("=" * 60)
    
    df = aggregate_csv_results(run_dirs, "ablation_results.csv")
    
    # Get converged stats (last 10 episodes)
    max_ep = df['episode'].max()
    df_final = df[df['episode'] >= max_ep - 10]
    
    stats = df_final.groupby('variant').agg({
        'mean_fidelity': ['mean', 'std'],
        'mean_completion_time': ['mean', 'std']
    }).reset_index()
    
    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\caption{Ablation Study: Impact of Error-Awareness}")
    latex.append(r"\begin{tabular}{lcc}")
    latex.append(r"\toprule")
    latex.append(r"\textbf{Configuration} & \textbf{Fidelity} & \textbf{Time (s)} \\")
    latex.append(r"\midrule")
    
    for _, row in stats.iterrows():
        name = row['variant'][0] if isinstance(row['variant'], tuple) else row['variant']
        name = name.replace("WithErrorFeatures", "Full Model").replace("NoErrorFeatures", "w/o Error State")
        fid = f"{row['mean_fidelity']['mean']:.4f} $\pm$ {row['mean_fidelity']['std']:.4f}"
        time = f"{row['mean_completion_time']['mean']:.1f} $\pm$ {row['mean_completion_time']['std']:.1f}"
        latex.append(f"{name} & {fid} & {time} \\\\")
        
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    
    print("\n".join(latex))
    return "\n".join(latex)

# =========================================================
# TABLE 3: Federated Learning
# =========================================================
def generate_federated_table(run_dirs):
    print("\n" + "=" * 60)
    print("TABLE 3: Federated Learning Results")
    print("=" * 60)
    
    df = aggregate_csv_results(run_dirs, "federated_results.csv")
    
    # Get final round performance
    max_round = df['fed_round'].max()
    df_final = df[df['fed_round'] == max_round]
    
    stats = df_final.groupby('method').agg({
        'mean_fidelity': ['mean', 'std'],
        'mean_completion_time': ['mean', 'std']
    }).reset_index()
    
    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\caption{Federated Aggregation Strategy Comparison}")
    latex.append(r"\begin{tabular}{lcc}")
    latex.append(r"\toprule")
    latex.append(r"\textbf{Strategy} & \textbf{Final Fidelity} & \textbf{Convergence Time (s)} \\")
    latex.append(r"\midrule")
    
    for _, row in stats.iterrows():
        name = row['method'][0] if isinstance(row['method'], tuple) else row['method']
        fid = f"{row['mean_fidelity']['mean']:.4f}"
        time = f"{row['mean_completion_time']['mean']:.1f}"
        latex.append(f"{name} & {fid} & {time} \\\\")
        
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    
    print("\n".join(latex))
    return "\n".join(latex)

# =========================================================
# FIGURE 1: Representative Training Curve
# =========================================================
def generate_training_curve(run_dirs):
    print("\nGenerating training curve figure...")
    
    # Load all rainbow DQN results
    df = aggregate_csv_results(run_dirs, "rainbow_dqn_results.csv")
    
    # Calculate mean and std across all runs for each episode
    grouped = df.groupby('episode').agg({
        'mean_reward': ['mean', 'std'],
        'mean_fidelity': ['mean', 'std']
    }).reset_index()
    
    episodes = grouped['episode']
    mean_reward = grouped['mean_reward']['mean']
    std_reward = grouped['mean_reward']['std']
    mean_fidelity = grouped['mean_fidelity']['mean']
    std_fidelity = grouped['mean_fidelity']['std']
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    # Plot Reward
    color = 'tab:blue'
    ax1.set_xlabel('Training Episode')
    ax1.set_ylabel('Average Reward', color=color)
    ax1.plot(episodes, mean_reward, color=color, label='Reward')
    ax1.fill_between(episodes, mean_reward - std_reward, mean_reward + std_reward, color=color, alpha=0.2)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Twin axis for Fidelity
    ax2 = ax1.twinx() 
    color = 'tab:green'
    ax2.set_ylabel('Circuit Fidelity', color=color) 
    ax2.plot(episodes, mean_fidelity, color=color, label='Fidelity', linestyle='--')
    ax2.fill_between(episodes, mean_fidelity - std_fidelity, mean_fidelity + std_fidelity, color=color, alpha=0.2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(f"FedMO-DRLQ Training Convergence (Avg over {len(run_dirs)} datasets)")
    fig.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, "training_curve.pdf")
    plt.savefig(save_path, dpi=300)
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300)
    print(f"Saved figure to {save_path}")
    plt.close()

# =========================================================
# PATCH 2: Statistical Analysis Functions
# =========================================================
def compute_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """Compute mean and 95% confidence interval."""
    n = len(data)
    if n < 2:
        return np.mean(data), np.mean(data), np.mean(data)
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - h, mean + h

def paired_statistical_test(drl_values: List[float], baseline_values: List[float]) -> Tuple[float, str]:
    """
    Perform paired Wilcoxon signed-rank test.
    Returns: (p_value, significance_string)
    """
    if len(drl_values) < 5:
        return 1.0, "insufficient data"
    
    try:
        stat, p_value = stats.wilcoxon(drl_values, baseline_values)
    except ValueError:
        return 1.0, "no difference"
    
    if p_value < 0.001:
        sig = "***"
    elif p_value < 0.01:
        sig = "**"
    elif p_value < 0.05:
        sig = "*"
    else:
        sig = "n.s."
    
    return p_value, sig

def generate_statistical_table(run_dirs: List[str]) -> str:
    """Generate LaTeX table with statistical analysis."""
    print("\n" + "=" * 60)
    print("TABLE: Statistical Comparison")
    print("=" * 60)
    
    # Collect paired results from each run
    drl_fidelity = []
    drl_time = []
    heuristic_results = {
        'Random': {'fidelity': [], 'time': []},
        'RoundRobin': {'fidelity': [], 'time': []},
        'MinCompletionTime': {'fidelity': [], 'time': []},
        'MinError': {'fidelity': [], 'time': []},
        'BestFidelity': {'fidelity': [], 'time': []}
    }
    
    for run_dir in run_dirs:
        summary_path = os.path.join(run_dir, "summary.json")
        if not os.path.exists(summary_path):
            continue
        
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        # Skip runs with insufficient training
        if summary.get('drl', {}).get('total_episodes', 0) < 5000:
            continue
        
        drl_fidelity.append(summary['drl']['final_mean_fidelity'])
        drl_time.append(summary['drl']['final_mean_completion_time'])
        
        for name in heuristic_results.keys():
            if name in summary.get('heuristics', {}):
                heuristic_results[name]['fidelity'].append(summary['heuristics'][name]['mean_fidelity'])
                heuristic_results[name]['time'].append(summary['heuristics'][name]['mean_completion_time'])
    
    n_runs = len(drl_fidelity)
    print(f"Statistical analysis based on {n_runs} runs")
    
    # Generate table
    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Performance comparison with statistical significance. " + 
                 f"Results aggregated over {n_runs} experimental runs. " +
                 r"$^{***}p<0.001$, $^{**}p<0.01$, $^{*}p<0.05$.}")
    latex.append(r"\label{tab:statistical_comparison}")
    latex.append(r"\begin{tabular}{lcccc}")
    latex.append(r"\toprule")
    latex.append(r"\textbf{Method} & \textbf{Fidelity} & \textbf{Time (s)} & \textbf{p-val (Fid)} & \textbf{p-val (Time)} \\")
    latex.append(r"\midrule")
    
    # DRL row
    if drl_fidelity:
        fid_mean, fid_lo, fid_hi = compute_confidence_interval(drl_fidelity)
        time_mean, time_lo, time_hi = compute_confidence_interval(drl_time)
        latex.append(f"\\textbf{{FedMO-DRLQ}} & \\textbf{{{fid_mean:.4f}}} $\\pm$ {(fid_hi-fid_lo)/2:.4f} & "
                     f"\\textbf{{{time_mean:.0f}}} $\\pm$ {(time_hi-time_lo)/2:.0f} & - & - \\\\")
        latex.append(r"\midrule")
    
    # Heuristic rows
    for name, data in heuristic_results.items():
        if len(data['fidelity']) == 0:
            continue
        
        fid_mean, fid_lo, fid_hi = compute_confidence_interval(data['fidelity'])
        time_mean, time_lo, time_hi = compute_confidence_interval(data['time'])
        
        p_fid, sig_fid = paired_statistical_test(drl_fidelity, data['fidelity'])
        p_time, sig_time = paired_statistical_test(drl_time, data['time'])
        
        latex.append(f"{name} & {fid_mean:.4f} $\\pm$ {(fid_hi-fid_lo)/2:.4f} & "
                     f"{time_mean:.0f} $\\pm$ {(time_hi-time_lo)/2:.0f} & "
                     f"{p_fid:.3f}$^{{{sig_fid}}}$ & {p_time:.3f}$^{{{sig_time}}}$ \\\\")
    
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    
    print("\n".join(latex))
    return "\n".join(latex)

# =========================================================
# PATCH 3: Publication-Quality Learning Curves
# =========================================================
def generate_learning_curves(run_dirs: List[str], output_dir: str = OUTPUT_DIR):
    """Generate publication-quality learning curves with confidence bands."""
    print("\nGenerating learning curves with baselines...")
    
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 150
    })
    
    # Load training histories
    all_rewards = []
    all_fidelities = []
    all_times = []
    
    for run_dir in run_dirs:
        drl_path = os.path.join(run_dir, "rainbow_dqn_results.csv")
        if not os.path.exists(drl_path):
            continue
        
        df = pd.read_csv(drl_path)
        if len(df) < 5000:  # Skip short runs
            continue
        
        all_rewards.append(df['mean_reward'].values)
        all_fidelities.append(df['mean_fidelity'].values)
        all_times.append(df['mean_completion_time'].values)
    
    if not all_rewards:
        print("No valid training histories found")
        return
    
    # Align to minimum length
    min_len = min(len(r) for r in all_rewards)
    rewards = np.array([r[:min_len] for r in all_rewards])
    fidelities = np.array([f[:min_len] for f in all_fidelities])
    times = np.array([t[:min_len] for t in all_times])
    
    episodes = np.arange(min_len)
    
    # Smoothing function
    def smooth(data, window=100):
        kernel = np.ones(window) / window
        return np.array([np.convolve(d, kernel, mode='valid') for d in data])
    
    rewards_smooth = smooth(rewards)
    fidelities_smooth = smooth(fidelities)
    times_smooth = smooth(times)
    
    ep_smooth = episodes[99:]  # Adjust for smoothing window
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Plot 1: Multi-Objective Reward
    mean_r = np.mean(rewards_smooth, axis=0)
    std_r = np.std(rewards_smooth, axis=0)
    axes[0].plot(ep_smooth, mean_r, 'b-', linewidth=2, label='FedMO-DRLQ')
    axes[0].fill_between(ep_smooth, mean_r - std_r, mean_r + std_r, alpha=0.3, color='blue')
    axes[0].set_xlabel('Training Episode')
    axes[0].set_ylabel('Multi-Objective Reward')
    axes[0].set_title('(a) Training Progress')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Fidelity with baselines
    mean_f = np.mean(fidelities_smooth, axis=0)
    std_f = np.std(fidelities_smooth, axis=0)
    axes[1].plot(ep_smooth, mean_f, 'g-', linewidth=2, label='FedMO-DRLQ')
    axes[1].fill_between(ep_smooth, mean_f - std_f, mean_f + std_f, alpha=0.3, color='green')
    axes[1].axhline(y=0.154, color='red', linestyle='--', linewidth=1.5, label='BestFidelity')
    axes[1].axhline(y=0.141, color='orange', linestyle='--', linewidth=1.5, label='MinCompletionTime')
    axes[1].set_xlabel('Training Episode')
    axes[1].set_ylabel('Circuit Fidelity')
    axes[1].set_title('(b) Fidelity Convergence')
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Completion Time with baselines
    mean_t = np.mean(times_smooth, axis=0)
    std_t = np.std(times_smooth, axis=0)
    axes[2].plot(ep_smooth, mean_t, 'purple', linewidth=2, label='FedMO-DRLQ')
    axes[2].fill_between(ep_smooth, mean_t - std_t, mean_t + std_t, alpha=0.3, color='purple')
    axes[2].axhline(y=14006, color='orange', linestyle='--', linewidth=1.5, label='MinCompletionTime')
    axes[2].axhline(y=20294, color='red', linestyle='--', linewidth=1.5, label='BestFidelity')
    axes[2].set_xlabel('Training Episode')
    axes[2].set_ylabel('Completion Time (s)')
    axes[2].set_title('(c) Time Convergence')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(os.path.join(output_dir, "learning_curves.pdf"), 
                dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(os.path.join(output_dir, "learning_curves.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Learning curves saved to {output_dir}")

# =========================================================
# DRL Algorithm Comparison Curves (Rainbow vs PPO vs Standard DQN)
# =========================================================
def generate_learning_curves_all_drl(run_dirs: List[str], output_dir: str = OUTPUT_DIR):
    """Generate learning curves comparing all DRL algorithms."""
    print("\nGenerating DRL algorithm comparison curves...")
    
    algorithms = {
        'Rainbow DQN (Ours)': ('rainbow_dqn_results.csv', 'blue', '-'),
        'PPO': ('ppo_results.csv', 'green', '--'),
        'Standard DQN': ('standard_dqn_results.csv', 'orange', ':')
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    for alg_name, (filename, color, linestyle) in algorithms.items():
        all_rewards = []
        all_fidelities = []
        all_times = []
        
        for run_dir in run_dirs:
            path = os.path.join(run_dir, filename)
            if os.path.exists(path):
                df = pd.read_csv(path)
                if len(df) >= 1000:
                    all_rewards.append(df['mean_reward'].values)
                    all_fidelities.append(df['mean_fidelity'].values)
                    all_times.append(df['mean_completion_time'].values)
        
        if not all_rewards:
            continue
        
        # Align and smooth
        min_len = min(len(r) for r in all_rewards)
        rewards = np.array([r[:min_len] for r in all_rewards])
        fidelities = np.array([f[:min_len] for f in all_fidelities])
        times = np.array([t[:min_len] for t in all_times])
        
        window = 100
        def smooth(data):
            return np.array([np.convolve(d, np.ones(window)/window, mode='valid') for d in data])
        
        rewards_s = smooth(rewards)
        fidelities_s = smooth(fidelities)
        times_s = smooth(times)
        
        episodes = np.arange(min_len - window + 1)
        
        # Plot reward
        mean_r = np.mean(rewards_s, axis=0)
        std_r = np.std(rewards_s, axis=0)
        axes[0].plot(episodes, mean_r, color=color, linestyle=linestyle, linewidth=2, label=alg_name)
        axes[0].fill_between(episodes, mean_r - std_r, mean_r + std_r, alpha=0.2, color=color)
        
        # Plot fidelity
        mean_f = np.mean(fidelities_s, axis=0)
        std_f = np.std(fidelities_s, axis=0)
        axes[1].plot(episodes, mean_f, color=color, linestyle=linestyle, linewidth=2, label=alg_name)
        axes[1].fill_between(episodes, mean_f - std_f, mean_f + std_f, alpha=0.2, color=color)
        
        # Plot time
        mean_t = np.mean(times_s, axis=0)
        std_t = np.std(times_s, axis=0)
        axes[2].plot(episodes, mean_t, color=color, linestyle=linestyle, linewidth=2, label=alg_name)
        axes[2].fill_between(episodes, mean_t - std_t, mean_t + std_t, alpha=0.2, color=color)
    
    # Add heuristic baselines
    axes[1].axhline(y=0.154, color='red', linestyle='-.', linewidth=1, label='BestFidelity')
    axes[2].axhline(y=14006, color='red', linestyle='-.', linewidth=1, label='MinCompletionTime')
    
    # Labels
    axes[0].set_xlabel('Training Episode')
    axes[0].set_ylabel('Multi-Objective Reward')
    axes[0].set_title('(a) Training Progress')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Training Episode')
    axes[1].set_ylabel('Circuit Fidelity')
    axes[1].set_title('(b) Fidelity Convergence')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Training Episode')
    axes[2].set_ylabel('Completion Time (s)')
    axes[2].set_title('(c) Time Convergence')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "learning_curves_all_drl.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "learning_curves_all_drl.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"DRL comparison learning curves saved to {output_dir}")

# =========================================================
# PATCH 6: Pareto Front Visualization
# =========================================================
def generate_pareto_front(run_dirs: List[str], output_dir: str = OUTPUT_DIR):
    """Generate Pareto front visualization for multi-objective results."""
    print("\nGenerating Pareto front visualization...")
    
    # Load multi-objective results
    all_mo_results = []
    for run_dir in run_dirs:
        mo_path = os.path.join(run_dir, "multi_objective_results.csv")
        if os.path.exists(mo_path):
            df = pd.read_csv(mo_path)
            # Get final performance for each configuration
            if 'config_name' in df.columns:
                for config_name in df['config_name'].unique():
                    cfg_data = df[df['config_name'] == config_name].tail(100)
                    all_mo_results.append({
                        'config': config_name,
                        'fidelity': cfg_data['mean_fidelity'].mean(),
                        'time': cfg_data['mean_completion_time'].mean(),
                        'run': run_dir
                    })
    
    if not all_mo_results:
        print("No multi-objective results found")
        return
    
    mo_df = pd.DataFrame(all_mo_results)
    
    # Aggregate by configuration
    summary = mo_df.groupby('config').agg({
        'fidelity': ['mean', 'std'],
        'time': ['mean', 'std']
    }).reset_index()
    
    # Create Pareto front plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = {
        'TimeFocused': 'blue',
        'FidelityFocused': 'green', 
        'Balanced': 'orange',
        'Chebyshev': 'red',
        'ConstraintBased': 'purple'
    }
    
    markers = {
        'TimeFocused': 'o',
        'FidelityFocused': 's',
        'Balanced': '^',
        'Chebyshev': 'D',
        'ConstraintBased': 'v'
    }
    
    for idx in range(len(summary)):
        config = summary.iloc[idx]['config']
        fid_mean = summary.iloc[idx][('fidelity', 'mean')]
        fid_std = summary.iloc[idx][('fidelity', 'std')]
        time_mean = summary.iloc[idx][('time', 'mean')]
        time_std = summary.iloc[idx][('time', 'std')]
        
        ax.errorbar(time_mean, fid_mean, 
                    xerr=time_std, yerr=fid_std,
                    fmt=markers.get(config, 'o'),
                    color=colors.get(config, 'gray'),
                    markersize=10, capsize=5,
                    label=config)
    
    ax.set_xlabel('Completion Time (s)', fontsize=12)
    ax.set_ylabel('Circuit Fidelity', fontsize=12)
    ax.set_title('Multi-Objective Pareto Front', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pareto_front.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "pareto_front.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Pareto front saved to {output_dir}")

# =========================================================
# MAIN
# =========================================================
def main():
    print("FedMO-DRLQ Results Generator for Q1 Journal")
    print("=" * 50)
    
    try:
        # 1. Find all relevant run directories
        run_dirs = get_all_run_dirs()
        
        # 2. Generate LaTeX tables
        all_latex = []
        all_latex.append("% Generated by generate_paper_results.py")
        all_latex.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        all_latex.append("")
        
        all_latex.append(generate_baseline_comparison_table(run_dirs))
        all_latex.append("")
        all_latex.append(generate_ablation_table(run_dirs))
        all_latex.append("")
        all_latex.append(generate_federated_table(run_dirs))
        all_latex.append("")
        
        # 3. Generate statistical analysis table (PATCH 2)
        try:
            all_latex.append(generate_statistical_table(run_dirs))
        except Exception as e:
            print(f"Skipping statistical table: {e}")
        
        # 4. Save LaTeX
        with open(os.path.join(OUTPUT_DIR, "all_tables.tex"), "w") as f:
            f.write("\n\n".join(all_latex))
        print(f"\nSaved LaTeX tables to {os.path.join(OUTPUT_DIR, 'all_tables.tex')}")
        
        # 5. Generate Figures
        try:
            generate_training_curve(run_dirs)
        except Exception as e:
            print(f"Skipping basic training curve: {e}")
        
        # 6. Generate publication-quality learning curves (PATCH 3)
        try:
            generate_learning_curves(run_dirs)
        except Exception as e:
            print(f"Skipping learning curves: {e}")
        
        # 7. Generate DRL algorithm comparison curves
        try:
            generate_learning_curves_all_drl(run_dirs)
        except Exception as e:
            print(f"Skipping DRL comparison curves: {e}")
        
        # 8. Generate Pareto front visualization (PATCH 6)
        try:
            generate_pareto_front(run_dirs)
        except Exception as e:
            print(f"Skipping Pareto front: {e}")
        
        print("\n" + "=" * 50)
        print("Paper results generated successfully!")
        print(f"Output directory: {OUTPUT_DIR}")
        print("=" * 50)
            
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        print("Please ensure you have run 'experiments/run_all_experiments.py' first.")

if __name__ == "__main__":
    main()