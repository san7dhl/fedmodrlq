# benchmark_comparison.py
from fedmo_drlq import (
    FedMODRLQConfig,
    FedMODRLQEnv,
    BenchmarkRunner,
    RainbowDQNAgent
)

config = FedMODRLQConfig()

# Create benchmark runner
runner = BenchmarkRunner(
    env_factory=lambda: FedMODRLQEnv(config=config)
)

# Run all baseline algorithms
print("Running baseline benchmarks...")
results = runner.run_all_baselines(num_episodes=10)

# Print comparison
runner.print_summary()

# Save results
runner.save_results("benchmark_results.json")