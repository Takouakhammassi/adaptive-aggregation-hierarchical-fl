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

NUM_CLIENTS, NUM_EDGES, CLOUD_ROUNDS, ALPHA = 30, 3, 30, 0.5
SCENARIOS = ["high", "moderate"]
SEEDS = [0, 1, 2]
OUT = "results/multiseed.json"

VARIANTS = {
    "naive_k2": lambda: NaiveController(2),
    "naive_k3": lambda: NaiveController(3),
    "decay_k2": lambda: AdaptiveController(2, decay="adaptive", anchor=True, adapt_freq=False),
    "decay_k3": lambda: AdaptiveController(3, decay="adaptive", anchor=True, adapt_freq=False),
    "full": lambda: AdaptiveController(2, decay="adaptive", anchor=True, adapt_freq=True),
}


def load_seed0():
    base = json.load(open("results/dynamic_baseline.json"))
    abl = json.load(open("results/ablation.json"))
    full = json.load(open("results/dynamic_adaptive.json"))
    res = {}
    for s in SCENARIOS:
        res[f"{s}/naive_k2/0"] = base[s]
        res[f"{s}/naive_k3/0"] = abl[f"{s}/naive_k3"]
        res[f"{s}/decay_k2/0"] = abl[f"{s}/decay_seule"]
        res[f"{s}/full/0"] = full[s]
    return res


device = get_device()
train_set, test_set = load_fashion_mnist()
labels = train_set.targets.numpy()
test_loader = DataLoader(test_set, batch_size=256)
parts_cache = {s: dirichlet_partition(labels, NUM_CLIENTS, ALPHA, seed=s) for s in SEEDS}

results = json.load(open(OUT)) if os.path.exists(OUT) else load_seed0()

for seed in SEEDS:
    for scen in SCENARIOS:
        for name, make in VARIANTS.items():
            key = f"{scen}/{name}/{seed}"
            if key in results:
                continue
            print(f"\n=== {key} ===")
            topo = DynamicTopology(NUM_CLIENTS, NUM_EDGES, scen, seed=seed)
            controllers = [make() for _ in range(NUM_EDGES)]
            results[key] = run_hierarchical_dynamic(
                train_set, test_loader, parts_cache[seed], topo, controllers,
                cloud_rounds=CLOUD_ROUNDS, seed=seed, device=device)
            with open(OUT, "w") as f:
                json.dump(results, f, indent=2)


def runs_of(scen, name):
    return [results[f"{scen}/{name}/{s}"] for s in SEEDS if f"{scen}/{name}/{s}" in results]


def round_to(hist, target=0.85):
    return next((h["round"] for h in hist if h["acc"] >= target), CLOUD_ROUNDS + 1)


for scen in SCENARIOS:
    print(f"\n--- {scen} (moyenne ± écart-type sur les seeds ; 31 = 85% jamais atteint) ---")
    print("{:<10} {:>6} {:>16} {:>16} {:>14}".format(
        "variante", "seeds", "acc (5 der.)", "round à 85%", "entraînements"))
    for name in VARIANTS:
        runs = runs_of(scen, name)
        if not runs:
            continue
        acc = [np.mean([h["acc"] for h in r[-5:]]) for r in runs]
        rd = [round_to(r) for r in runs]
        cu = [r[-1]["client_updates"] for r in runs]
        print("{:<10} {:>6} {:>8.3f} ± {:.3f} {:>9.1f} ± {:.1f} {:>14.0f}".format(
            name, len(runs), np.mean(acc), np.std(acc), np.mean(rd), np.std(rd), np.mean(cu)))

fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
for col, scen in enumerate(SCENARIOS):
    for name in VARIANTS:
        runs = runs_of(scen, name)
        if not runs:
            continue
        accs = np.array([[h["acc"] for h in r] for r in runs])
        cus = np.array([[h["client_updates"] for h in r] for r in runs]).mean(axis=0)
        m, sd = accs.mean(axis=0), accs.std(axis=0)
        x_round = np.arange(1, CLOUD_ROUNDS + 1)
        axes[0, col].plot(x_round, m, label=name)
        axes[0, col].fill_between(x_round, m - sd, m + sd, alpha=0.10)
        axes[1, col].plot(cus, m, label=name)
        axes[1, col].fill_between(cus, m - sd, m + sd, alpha=0.10)
    axes[0, col].set_title(f"Scénario : {scen}")
    axes[0, col].set_xlabel("Cloud round")
    axes[1, col].set_xlabel("Entraînements clients cumulés (moyenne)")
    axes[0, col].grid(alpha=0.3)
    axes[1, col].grid(alpha=0.3)
axes[0, 0].set_ylabel("Précision (test)")
axes[1, 0].set_ylabel("Précision (test)")
axes[0, 0].legend()
plt.tight_layout()
plt.savefig("results/multiseed.png", dpi=150)
print("Figure sauvegardée : results/multiseed.png")