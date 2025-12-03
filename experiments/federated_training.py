# federated_training.py
from fedmo_drlq import (
    FedMODRLQConfig,
    FedMODRLQEnv,
    RainbowDQNAgent,
    FederatedServer,
    FederatedClient
)

config = FedMODRLQConfig()
NUM_CLIENTS = 3  # Simulating 3 quantum datacenters

# Create local agents for each datacenter
clients = []
for i in range(NUM_CLIENTS):
    env = FedMODRLQEnv(config=config, agent_id=i)
    agent = RainbowDQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )
    clients.append((env, agent))
    print(f"✓ Client {i} created (Datacenter {i+1})")

# Create federated server
server = FederatedServer(
    global_model=clients[0][1].online_network,
    config=config.federated
)

print(f"\n✅ Federated setup complete with {NUM_CLIENTS} clients")
print("Ready for federated training!")