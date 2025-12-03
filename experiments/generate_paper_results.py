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
# MAIN
# =========================================================
def main():
    print("FedMO-DRLQ Results Generator")
    print("----------------------------")
    
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
        
        # 3. Save LaTeX
        with open(os.path.join(OUTPUT_DIR, "all_tables.tex"), "w") as f:
            f.write("\n\n".join(all_latex))
        print(f"\nSaved LaTeX tables to {os.path.join(OUTPUT_DIR, 'all_tables.tex')}")
        
        # 4. Generate Figures
        try:
            generate_training_curve(run_dirs)
        except Exception as e:
            print(f"Skipping figure generation due to error: {e}")
            
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        print("Please ensure you have run 'experiments/run_all_experiments.py' first.")

if __name__ == "__main__":
    main()