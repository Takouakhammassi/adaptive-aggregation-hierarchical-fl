import math

from src.federated import weighted_average


class NaiveController:

    def __init__(self, base_edge_rounds: int = 2):
        self.base = base_edge_rounds

    def edge_rounds(self):
        return self.base

    def observe(self, target_m, n_fresh, stale_ages):
        pass 

    def aggregate(self, fresh, stale, current_w):
        updates = list(fresh) + [(w, n) for (w, n, _) in stale]
        if not updates:
            return None
        return weighted_average(updates)


class AdaptiveController(NaiveController):

    def __init__(self, base_edge_rounds=2, max_edge_rounds=4, decay="adaptive",
                 anchor=True, adapt_freq=True, fixed_tau=2.0, beta=0.3):
        super().__init__(base_edge_rounds)
        self.max_rounds = max_edge_rounds
        self.decay = decay
        self.anchor = anchor
        self.adapt_freq = adapt_freq
        self.fixed_tau = fixed_tau
        self.beta = beta                
        self.fresh_ratio = 1.0          
        self.mean_staleness = 1.0       

    def observe(self, target_m, n_fresh, stale_ages):
        ratio = min(1.0, n_fresh / max(1, target_m))
        self.fresh_ratio = (1 - self.beta) * self.fresh_ratio + self.beta * ratio
        for s in stale_ages:
            self.mean_staleness = (1 - self.beta) * self.mean_staleness + self.beta * s

    def edge_rounds(self):
        if not self.adapt_freq:
            return self.base
        k = round(self.base / max(self.fresh_ratio, 0.25))
        return int(min(self.max_rounds, max(self.base, k)))

    def stale_weight(self, s):
        if self.decay == "none":
            return 1.0
        if self.decay == "fixed":
            tau = self.fixed_tau
        else:  
            tau = min(4.0, max(1.0, self.mean_staleness))
        return math.exp(-s / tau)

    def aggregate(self, fresh, stale, current_w):
        updates = list(fresh) + [(w, n * self.stale_weight(s)) for (w, n, s) in stale]
        if not updates:
            return None
        if self.anchor and not fresh:
            updates.append((current_w, sum(n for _, n, _ in stale)))
        return weighted_average(updates)