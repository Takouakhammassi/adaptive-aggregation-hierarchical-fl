import json

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.data import dirichlet_partition, load_fashion_mnist
from src.federated import run_fedavg
from src.training import get_device

NUM_CLIENTS = 30
ROUNDS = 30
CENTRALIZED_ACC = 0.887  

device = get_device()
train_set, test_set = load_fashion_mnist()
labels = train_set.targets.numpy()
test_loader = DataLoader(test_set, batch_size=256)

results = {}
for alpha in [100, 0.5, 0.1]:
    print(f"\n=== FedAvg, alpha = {alpha} ===")
    parts = dirichlet_partition(labels, NUM_CLIENTS, alpha, seed=0)
    results[str(alpha)] = run_fedavg(
        train_set, test_loader, parts, rounds=ROUNDS, frac=0.3, seed=0, device=device
    )

with open("results/fedavg_baseline.json", "w") as f:
    json.dump(results, f, indent=2)

plt.figure(figsize=(7, 4.5))
for alpha, hist in results.items():
    plt.plot([h["round"] for h in hist], [h["acc"] for h in hist], label=f"α = {alpha}")
plt.axhline(CENTRALIZED_ACC, color="gray", linestyle="--", label="Centralisé")
plt.xlabel("Round")
plt.ylabel("Précision (test)")
plt.title("FedAvg selon l'hétérogénéité des données")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/fedavg_alpha.png", dpi=150)
print("Figure sauvegardée dans results/fedavg_alpha.png")