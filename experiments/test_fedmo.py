# test_fedmo.py
from fedmo_drlq import (
    FedMODRLQConfig,
    FedMODRLQEnv,
    RainbowDQNAgent,
    QNodeErrorMetrics,
    FidelityEstimator
)

print("=" * 50)
print("FedMO-DRLQ Test")
print("=" * 50)

# 1. Test Configuration
config = FedMODRLQConfig()
print(f"✓ Config created")

# 2. Test Environment
env = FedMODRLQEnv(config=config)
print(f"✓ Environment created")
print(f"  - Observation space: {env.observation_space.shape}")
print(f"  - Action space: {env.action_space.n}")

# 3. Test Agent
agent = RainbowDQNAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    device="cpu"
)
print(f"✓ Agent created")

# 4. Quick test
obs, info = env.reset()
action = agent.select_action(obs)
next_obs, reward, done, _, info = env.step(action)
print(f"✓ Step executed: reward={reward:.4f}")

print("\n" + "=" * 50)
print("✅ ALL TESTS PASSED!")
print("=" * 50)