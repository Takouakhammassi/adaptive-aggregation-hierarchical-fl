import json

import numpy as np

res = json.load(open("results/multiseed.json"))
SEEDS = [0, 1, 2]
for scen in ["high", "moderate"]:
    print(f"\n--- {scen} ---")
    print("{:<10} {:>24} {:>12}".format("variante", "chutes >= 3 pts par run", "pire chute"))
    for name in ["naive_k2", "naive_k3", "decay_k2", "decay_k3", "full"]:
        counts, worst = [], 0.0
        for s in SEEDS:
            acc = np.array([h["acc"] for h in res[f"{scen}/{name}/{s}"]])
            drops = acc[:-1] - acc[1:]
            counts.append(int((drops >= 0.03).sum()))
            worst = max(worst, float(drops.max()))
        print("{:<10} {:>24.1f} {:>12.3f}".format(name, np.mean(counts), worst))