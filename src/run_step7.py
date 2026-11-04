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
CLOUD_ROUNDS = 30
ALPHA = 0.5
SCENARIOS = ["moderate", "high"]
OUT = "results/ablation.json"

VARIANTS = {
    "naive_k3": lambda: NaiveController(3),
    "freq_seule": lambda: AdaptiveController(2, decay="none", anchor=False, adapt_freq=True),
    "decay_seule": lambda: AdaptiveController(2, decay="adaptive", anchor=True, adapt_freq=False),
    "decay_fixe": lambda: AdaptiveController(2, decay="fixed", anchor=True, adapt_freq=True),
}
ORDER = ["naive_k2", "naive_k3", "freq_seule", "decay_seule", "decay_fixe", "full"]

device = get_device()
train_set, test_set = load_fashion_mnist()
labels = train_set.targets.numpy()
test_loader = DataLoader(test_set, batch_size=256)
parts = dirichlet_partition(labels, NUM_CLIENTS, ALPHA, seed=0)

results = json.load(open(OUT)) if os.path.exists(OUT) else {}

for scen in SCENARIOS:
    for name, make in VARIANTS.items():
        key = f"{scen}/{name}"
        if key in results:
            continue
        print(f"\n=== {key} ===")
        topo = DynamicTopology(NUM_CLIENTS, NUM_EDGES, scen, seed=0)
        controllers = [make() for _ in range(NUM_EDGES)]
        results[key] = run_hierarchical_dynamic(
            train_set, test_loader, parts, topo, controllers,
            cloud_rounds=CLOUD_ROUNDS, seed=0, device=device)
        with open(OUT, "w") as f:
            json.dump(results, f, indent=2)

naive2 = json.load(open("results/dynamic_baseline.json"))
full = json.load(open("results/dynamic_adaptive.json"))
allres = {}
for scen in SCENARIOS:
    allres[scen] = {"naive_k2": naive2[scen], "full": full[scen]}
    for name in VARIANTS:
        allres[scen][name] = results[f"{scen}/{name}"]


def acc_at_budget(hist, budget):
    ok = [h["acc"] for h in hist if h["client_updates"] <= budget]
    return np.mean(ok[-3:]) if ok else float("nan")


def round_to(hist, target=0.85):
    return next((h["round"] for h in hist if h["acc"] >= target), None)


for scen in SCENARIOS:
    budget = allres[scen]["naive_k2"][-1]["client_updates"]
    print(f"\n--- {scen} (budget = {budget} entraînements) ---")
    print("{:<12} {:>12} {:>12} {:>14} {:>14}".format(
        "variante", "acc (5 der.)", "round à 85%", "entraînements", "acc à budget"))
    for name in ORDER:
        h = allres[scen][name]
        print("{:<12} {:>12.3f} {:>12} {:>14} {:>14.3f}".format(
            name, np.mean([x["acc"] for x in h[-5:]]), str(round_to(h)),
            h[-1]["client_updates"], acc_at_budget(h, budget)))

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, scen in zip(axes, SCENARIOS):
    for name in ORDER:
        h = allres[scen][name]
        ax.plot([x["client_updates"] for x in h], [x["acc"] for x in h], label=name)
    ax.set_title(f"Scénario : {scen}")
    ax.set_xlabel("Entraînements clients cumulés")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Précision (test)")
axes[0].legend()
plt.tight_layout()
plt.savefig("results/ablation.png", dpi=150)
print("Figure sauvegardée : results/ablation.png")