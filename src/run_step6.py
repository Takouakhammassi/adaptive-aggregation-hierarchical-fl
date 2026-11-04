import json
import os

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from src.controllers import AdaptiveController, NaiveController
from src.data import dirichlet_partition, load_fashion_mnist
from src.hier_dynamic import run_hierarchical_dynamic
from src.topology import DynamicTopology
from src.training import get_device

NUM_CLIENTS, NUM_EDGES = 30, 3
CLOUD_ROUNDS, BASE_EDGE_ROUNDS = 30, 2
ALPHA = 0.5
SCENARIOS = ["static", "moderate", "high"]
RERUN_NAIVE = False
BASELINE_FILE = "results/dynamic_baseline.json"

device = get_device()
train_set, test_set = load_fashion_mnist()
labels = train_set.targets.numpy()
test_loader = DataLoader(test_set, batch_size=256)
parts = dirichlet_partition(labels, NUM_CLIENTS, ALPHA, seed=0)


def run(scenario, make_controller):
    topo = DynamicTopology(NUM_CLIENTS, NUM_EDGES, scenario, seed=0)
    controllers = [make_controller() for _ in range(NUM_EDGES)]
    return run_hierarchical_dynamic(train_set, test_loader, parts, topo, controllers, cloud_rounds=CLOUD_ROUNDS, seed=0, device=device)


# ---- baseline naïf ----
if RERUN_NAIVE or not os.path.exists(BASELINE_FILE):
    naive = {}
    for s in SCENARIOS:
        print(f"\n=== NAÏF, scénario = {s} ===")
        naive[s] = run(s, lambda: NaiveController(BASE_EDGE_ROUNDS))
    with open(BASELINE_FILE, "w") as f:
        json.dump(naive, f, indent=2)
else:
    with open(BASELINE_FILE) as f:
        naive = json.load(f)

# ---- mécanisme adaptatif ----
adaptive = {}
for s in SCENARIOS:
    print(f"\n=== ADAPTATIF, scénario = {s} ===")
    adaptive[s] = run(s, lambda: AdaptiveController(base_edge_rounds=BASE_EDGE_ROUNDS))
with open("results/dynamic_adaptive.json", "w") as f:
    json.dump(adaptive, f, indent=2)


# ---- résumé chiffré ----
def summarize(hist, target=0.85):
    accs = [h["acc"] for h in hist]
    reach = next((h["round"] for h in hist if h["acc"] >= target), None)
    return np.mean(accs[-5:]), reach, hist[-1]["client_updates"], hist[-1]["edge_cloud_msgs"]


print("\n{:<10} {:<10} {:>12} {:>14} {:>16} {:>10}".format(
    "scénario", "méthode", "acc (5 der.)", "round à 85%", "entraînements", "msgs Cloud"))
for s in SCENARIOS:
    for name, res in [("naïf", naive), ("adaptatif", adaptive)]:
        acc, reach, cu, msgs = summarize(res[s])
        print("{:<10} {:<10} {:>12.3f} {:>14} {:>16} {:>10}".format(s, name, acc, str(reach), cu, msgs))

# ---- figure 1 : naïf vs adaptatif ----
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
for col, s in enumerate(SCENARIOS):
    n_h, a_h = naive[s], adaptive[s]
    ax = axes[0, col]
    ax.plot([h["round"] for h in n_h], [h["acc"] for h in n_h], label="Naïf")
    ax.plot([h["round"] for h in a_h], [h["acc"] for h in a_h], label="Adaptatif")
    ax.set_title(f"Scénario : {s}")
    ax.set_xlabel("Cloud round")
    ax = axes[1, col]
    ax.plot([h["client_updates"] for h in n_h], [h["acc"] for h in n_h], label="Naïf")
    ax.plot([h["client_updates"] for h in a_h], [h["acc"] for h in a_h], label="Adaptatif")
    ax.set_xlabel("Entraînements clients cumulés")
    axes[0, col].grid(alpha=0.3)
    axes[1, col].grid(alpha=0.3)
axes[0, 0].set_ylabel("Précision (test)")
axes[1, 0].set_ylabel("Précision (test)")
axes[0, 0].legend()
plt.suptitle(f"Agrégation naïve vs adaptative (α = {ALPHA})")
plt.tight_layout()
plt.savefig("results/naive_vs_adaptive.png", dpi=150)

# ---- figure 2 : chaque région adapte sa fréquence ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, s in zip(axes, ["moderate", "high"]):
    ks = np.array([h["edge_rounds"] for h in adaptive[s]])
    for e in range(NUM_EDGES):
        ax.plot(range(1, CLOUD_ROUNDS + 1), ks[:, e], label=f"Région {e}")
    ax.set_title(f"Edge rounds par région ({s})")
    ax.set_xlabel("Cloud round")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Edge rounds")
axes[0].legend()
plt.tight_layout()
plt.savefig("results/adaptive_edge_rounds.png", dpi=150)
print("Figures sauvegardées dans results/")