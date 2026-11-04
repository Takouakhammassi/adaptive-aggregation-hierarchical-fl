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
SCENARIO, SEED = "high", 0
SIGMAS = [0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
OUT = "results/dp.json"
METHODS = {
    "naive_k3": lambda: NaiveController(3),
    "decay_k3": lambda: AdaptiveController(3, decay="adaptive", anchor=True, adapt_freq=False),
}

device = get_device()
train_set, test_set = load_fashion_mnist()
labels = train_set.targets.numpy()
test_loader = DataLoader(test_set, batch_size=256)
parts = dirichlet_partition(labels, NUM_CLIENTS, ALPHA, seed=SEED)


def run(make, rounds, dp_clip=None, dp_sigma=0.0):
    topo = DynamicTopology(NUM_CLIENTS, NUM_EDGES, SCENARIO, seed=SEED)
    controllers = [make() for _ in range(NUM_EDGES)]
    return run_hierarchical_dynamic(train_set, test_loader, parts, topo, controllers, cloud_rounds=rounds, seed=SEED, device=device, dp_clip=dp_clip, dp_sigma=dp_sigma)


def save(res):
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)


results = json.load(open(OUT)) if os.path.exists(OUT) else {}

if "clip" not in results:
    print("=== Calibration (6 rounds, sans DP) ===")
    calib = run(METHODS["decay_k3"], 6)
    results["clip"] = round(float(np.median([h["update_norm"] for h in calib])), 3)
    save(results)
CLIP = results["clip"]
print(f"\nSeuil de clipping C = {CLIP}")

for name, make in METHODS.items():
    for sigma in SIGMAS:
        key = f"{name}/{sigma}"
        if key in results:
            continue
        print(f"\n=== {key} (C = {CLIP}) ===")
        results[key] = run(make, CLOUD_ROUNDS, dp_clip=CLIP, dp_sigma=sigma)
        save(results)


def last5(h):
    return np.mean([x["acc"] for x in h[-5:]])


def round_to(h, target=0.85):
    return next((x["round"] for x in h if x["acc"] >= target), CLOUD_ROUNDS + 1)


print(f"\n--- {SCENARIO}, seed {SEED}, C = {CLIP} (31 = 85% jamais atteint) ---")
print("{:>8} | {:>18} | {:>18}".format("sigma", "naive_k3", "decay_k3"))
print("{:>8} | {:>9} {:>8} | {:>9} {:>8}".format("", "acc", "round85", "acc", "round85"))
for sigma in SIGMAS:
    hn, hd = results[f"naive_k3/{sigma}"], results[f"decay_k3/{sigma}"]
    print("{:>8} | {:>9.3f} {:>8} | {:>9.3f} {:>8}".format(
        sigma, last5(hn), round_to(hn), last5(hd), round_to(hd)))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for sigma in SIGMAS:
    h = results[f"decay_k3/{sigma}"]
    axes[0].plot([x["round"] for x in h], [x["acc"] for x in h], label=f"σ = {sigma}")
axes[0].set_title("decay_k3 : précision selon le niveau de bruit")
axes[0].set_xlabel("Cloud round")
axes[0].set_ylabel("Précision (test)")
axes[0].legend()
axes[0].grid(alpha=0.3)
for name in METHODS:
    axes[1].plot(range(len(SIGMAS)), [last5(results[f"{name}/{s}"]) for s in SIGMAS],
                 marker="o", label=name)
axes[1].set_xticks(range(len(SIGMAS)))
axes[1].set_xticklabels([str(s) for s in SIGMAS])
axes[1].set_xlabel("σ (noise multiplier)")
axes[1].set_ylabel("Précision (5 derniers rounds)")
axes[1].set_title("Compromis précision / bruit")
axes[1].set_ylim(0.0, 1.0)
axes[1].legend()
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/dp.png", dpi=150)
print("Figure sauvegardée : results/dp.png")