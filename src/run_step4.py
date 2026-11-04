import json

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.data import dirichlet_partition, load_fashion_mnist
from src.hierarchical import assign_clients_to_edges, run_hierarchical
from src.training import get_device

NUM_CLIENTS, NUM_EDGES = 30, 3
CLOUD_ROUNDS, EDGE_ROUNDS = 30, 2
FLAT_CLIENTS_PER_ROUND = int(0.3 * NUM_CLIENTS) 

device = get_device()
train_set, test_set = load_fashion_mnist()
labels = train_set.targets.numpy()
test_loader = DataLoader(test_set, batch_size=256)

with open("results/fedavg_baseline.json") as f:
    flat = json.load(f)

members = assign_clients_to_edges(NUM_CLIENTS, NUM_EDGES, seed=0)
print("Régions :", members)

hier = {}
for alpha in [100, 0.5, 0.1]:
    print(f"\n=== FedAvg hiérarchique, alpha = {alpha} ===")
    parts = dirichlet_partition(labels, NUM_CLIENTS, alpha, seed=0)
    hier[str(alpha)] = run_hierarchical(
        train_set, test_loader, parts, membership_fn=lambda t: members,
        cloud_rounds=CLOUD_ROUNDS, edge_rounds=EDGE_ROUNDS, seed=0, device=device,
    )

with open("results/hier_baseline.json", "w") as f:
    json.dump(hier, f, indent=2)

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
for col, alpha in enumerate(["100", "0.5", "0.1"]):
    f_hist, h_hist = flat[alpha], hier[alpha]
    # Ligne 1 : par round
    ax = axes[0, col]
    ax.plot([h["round"] for h in f_hist], [h["acc"] for h in f_hist], label="FedAvg plat")
    ax.plot([h["round"] for h in h_hist], [h["acc"] for h in h_hist], label="FedAvg hiérarchique")
    ax.set_title(f"α = {alpha}")
    ax.set_xlabel("Round (cloud)")
    
    ax = axes[1, col]
    ax.plot([h["round"] * FLAT_CLIENTS_PER_ROUND for h in f_hist], [h["acc"] for h in f_hist], label="FedAvg plat")
    ax.plot([h["client_updates"] for h in h_hist], [h["acc"] for h in h_hist], label="FedAvg hiérarchique")
    ax.set_xlabel("Entraînements clients cumulés")
    for r in (0, 1):
        axes[r, col].grid(alpha=0.3)
axes[0, 0].set_ylabel("Précision (test)")
axes[1, 0].set_ylabel("Précision (test)")
axes[0, 0].legend()
plt.suptitle("FedAvg plat vs hiérarchique")
plt.tight_layout()
plt.savefig("results/flat_vs_hier.png", dpi=150)
print("Figure sauvegardée dans results/flat_vs_hier.png")