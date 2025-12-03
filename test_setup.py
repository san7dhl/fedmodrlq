# test_setup.py
import sys
sys.path.insert(0, 'fedmo_package')

# Test imports
from fedmo_drlq.core import NISQDeviceMetrics, MultiObjectiveReward
from fedmo_drlq.agents import RainbowDQNAgent
from fedmo_drlq.federated import FederatedAggregator

print("✓ All imports successful!")

# Quick test
metrics = NISQDeviceMetrics(device_name="test", num_qubits=27)
print(f"✓ NISQDeviceMetrics created: {metrics.num_qubits} qubits")

reward_calc = MultiObjectiveReward()
r = reward_calc.compute_reward(completion_time=10.0, fidelity=0.9, energy=5000)
print(f"✓ MultiObjectiveReward working: reward = {r:.4f}")

print("\n✅ Setup complete! Ready to run experiments.")