import json

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from src.data import dirichlet_partition, load_fashion_mnist
from src.hier_dynamic import aggregate_naive, run_hierarchical_dynamic
from src.topology import SCENARIOS, DynamicTopology
from src.training import get_device

NUM_CLIENTS, NUM_EDGES = 30, 3
CLOUD_ROUNDS, EDGE_ROUNDS = 30, 2
ALPHA = 0.5
scenarios = list(SCENARIOS)  

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, name in zip(axes, scenarios):
    topo = DynamicTopology(NUM_CLIENTS, NUM_EDGES, name, seed=0)
    sizes = []
    for _ in range(CLOUD_ROUNDS):
        topo.step()
        sizes.append([len(topo.members(e)) for e in range(NUM_EDGES)])
    sizes = np.array(sizes)
    for e in range(NUM_EDGES):
        ax.plot(range(1, CLOUD_ROUNDS + 1), sizes[:, e], label=f"Région {e}")
    ax.set_title(f"Scénario : {name}")
    ax.set_xlabel("Cloud round")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Nombre de clients")
axes[0].legend()
plt.tight_layout()
plt.savefig("results/topology_dynamics.png", dpi=150)
print("Figure sauvegardée : results/topology_dynamics.png")

device = get_device()
train_set, test_set = load_fashion_mnist()
labels = train_set.targets.numpy()
test_loader = DataLoader(test_set, batch_size=256)
parts = dirichlet_partition(labels, NUM_CLIENTS, ALPHA, seed=0)

results = {}
for name in scenarios:
    print(f"\n=== Hiérarchique naïf, scénario = {name}, alpha = {ALPHA} ===")
    topo = DynamicTopology(NUM_CLIENTS, NUM_EDGES, name, seed=0)
    results[name] = run_hierarchical_dynamic(
        train_set, test_loader, parts, topo, aggregate_naive,
        cloud_rounds=CLOUD_ROUNDS, edge_rounds=EDGE_ROUNDS, seed=0, device=device,
    )

with open("results/dynamic_baseline.json", "w") as f:
    json.dump(results, f, indent=2)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for name, hist in results.items():
    rounds = [h["round"] for h in hist]
    axes[0].plot(rounds, [h["acc"] for h in hist], label=name)
    axes[1].plot(rounds, [h["stale_used"] for h in hist], label=name)
axes[0].set_title(f"Précision du baseline naïf (α = {ALPHA})")
axes[0].set_xlabel("Cloud round")
axes[0].set_ylabel("Précision (test)")
axes[1].set_title("Mises à jour en retard agrégées par round")
axes[1].set_xlabel("Cloud round")
for ax in axes:
    ax.grid(alpha=0.3)
    ax.legend()
plt.tight_layout()
plt.savefig("results/dynamic_baseline.png", dpi=150)
print("Figure sauvegardée : results/dynamic_baseline.png")