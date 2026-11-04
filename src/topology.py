from dataclasses import dataclass

import numpy as np


@dataclass
class RegionProfile:
    availability: float  
    p_slow: float        
    max_delay: int      


SCENARIOS = {
    "static": dict(p_move=0.00, profiles=[(1.0, 0.0, 0), (1.0, 0.0, 0), (1.0, 0.0, 0)]),
    "moderate": dict(p_move=0.05, profiles=[(0.9, 0.1, 2), (0.7, 0.3, 3), (0.5, 0.5, 4)]),
    "high": dict(p_move=0.15, profiles=[(0.8, 0.2, 2), (0.5, 0.5, 4), (0.3, 0.7, 6)]),
}


class DynamicTopology:

    def __init__(self, num_clients: int, num_edges: int, scenario: str, seed: int = 0):
        cfg = SCENARIOS[scenario]
        assert len(cfg["profiles"]) == num_edges, "un profil par région est nécessaire"
        self.rng = np.random.default_rng(seed) 
        self.num_edges = num_edges
        self.p_move = cfg["p_move"]
        self.profiles = [RegionProfile(*p) for p in cfg["profiles"]]

        self.region_of = np.empty(num_clients, dtype=int)
        perm = self.rng.permutation(num_clients)
        for e, chunk in enumerate(np.array_split(perm, num_edges)):
            self.region_of[chunk] = e

    def step(self):
        for c in range(len(self.region_of)):
            if self.rng.random() < self.p_move:
                others = [e for e in range(self.num_edges) if e != self.region_of[c]]
                self.region_of[c] = self.rng.choice(others)

    def members(self, e: int):
        return np.where(self.region_of == e)[0].tolist()