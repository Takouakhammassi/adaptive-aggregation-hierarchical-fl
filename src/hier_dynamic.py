import numpy as np
import torch

from src.federated import client_update, get_weights, set_weights, weighted_average
from src.model import SmallCNN
from src.privacy import add_noise, clip_update, update_norm
from src.training import evaluate


def run_hierarchical_dynamic(train_set, test_loader, client_indices, topology, controllers, cloud_rounds=30, frac=0.3, local_epochs=1, lr=0.05, batch_size=32, seed=0, device="cpu", dp_clip=None, dp_sigma=0.0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    noise_gen = torch.Generator().manual_seed(seed + 12345)  

    num_edges = topology.num_edges
    global_model = SmallCNN()
    pending = [[] for _ in range(num_edges)]  
    ticks = [0] * num_edges
    history = []
    client_updates_total = 0
    edge_cloud_msgs = 0

    for t in range(1, cloud_rounds + 1):
        topology.step()
        global_weights = get_weights(global_model)
        edge_results = []
        n_fresh = n_stale = stale_sum = 0
        rounds_used = []
        norms = []

        for e in range(num_edges):
            members = topology.members(e)
            if not members:
                rounds_used.append(0)
                continue
            profile = topology.profiles[e]
            ctrl = controllers[e]
            edge_w = global_weights
            k_e = ctrl.edge_rounds()
            rounds_used.append(k_e)

            for _ in range(k_e):
                ticks[e] += 1
                tick = ticks[e]
                target_m = max(1, int(frac * len(members)))

                online = [c for c in members if rng.random() < profile.availability]
                fresh = []
                if online:
                    chosen = rng.choice(online, size=min(len(online), target_m), replace=False)
                    for c in chosen:
                        w, n = client_update(edge_w, train_set, client_indices[c], local_epochs, lr, batch_size, device)
                        client_updates_total += 1
                        if dp_clip is not None:      
                            w, norm = clip_update(w, edge_w, dp_clip)
                        else:
                            norm = update_norm(w, edge_w)
                        norms.append(norm)
                        if rng.random() < profile.p_slow:
                            delay = int(rng.integers(1, max(1, profile.max_delay) + 1))
                            pending[e].append((w, n, tick, tick + delay))
                        else:
                            fresh.append((w, n))

                arrived = [(w, n, tick - base) for (w, n, base, arr) in pending[e] if arr <= tick]
                pending[e] = [p for p in pending[e] if p[3] > tick]

                ctrl.observe(target_m, len(fresh), [s for _, _, s in arrived])
                new_w = ctrl.aggregate(fresh, arrived, edge_w)
                if new_w is not None:
                    if dp_clip is not None and dp_sigma > 0:  
                        m_agg = len(fresh) + len(arrived)
                        new_w = add_noise(new_w, dp_sigma * dp_clip / m_agg, noise_gen)
                    edge_w = new_w

                n_fresh += len(fresh)
                n_stale += len(arrived)
                stale_sum += sum(s for _, _, s in arrived)

            n_edge = sum(len(client_indices[c]) for c in members)
            edge_results.append((edge_w, n_edge))
            edge_cloud_msgs += 1

        set_weights(global_model, weighted_average(edge_results))
        loss, acc = evaluate(global_model, test_loader, device)
        history.append({
            "round": t, "loss": loss, "acc": acc,
            "client_updates": client_updates_total, "edge_cloud_msgs": edge_cloud_msgs,
            "fresh_used": n_fresh, "stale_used": n_stale,
            "avg_staleness": stale_sum / n_stale if n_stale else 0.0,
            "edge_rounds": rounds_used,
            "update_norm": float(np.mean(norms)) if norms else 0.0,
        })
        print(f"Cloud round {t:02d} | acc {acc:.3f} | fraîches {n_fresh} | "
              f"en retard {n_stale} | norme moy. {history[-1]['update_norm']:.3f}")

    return history