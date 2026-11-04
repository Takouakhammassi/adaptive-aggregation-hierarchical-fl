import numpy as np
import torch

from src.federated import client_update, get_weights, set_weights, weighted_average
from src.model import SmallCNN
from src.training import evaluate


def assign_clients_to_edges(num_clients, num_edges, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_clients)
    return [sorted(chunk.tolist()) for chunk in np.array_split(perm, num_edges)]


def edge_round(edge_weights, members, train_set, client_indices, rng, frac, epochs, lr, batch_size, device):
    m = max(1, int(frac * len(members)))
    chosen = rng.choice(members, size=m, replace=False)
    updates = [
        client_update(edge_weights, train_set, client_indices[c], epochs, lr, batch_size, device)
        for c in chosen
    ]
    return weighted_average(updates), len(chosen)


def run_hierarchical(train_set, test_loader, client_indices, membership_fn, cloud_rounds=30, edge_rounds=2, frac=0.3, local_epochs=1, lr=0.05, batch_size=32, seed=0, device="cpu"):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)  

    global_model = SmallCNN()
    history = []
    client_updates_total = 0   
    edge_cloud_msgs = 0        

    for t in range(1, cloud_rounds + 1):
        members_per_edge = membership_fn(t)
        global_weights = get_weights(global_model)
        edge_results = []

        for members in members_per_edge:
            if len(members) == 0:
                continue
            edge_w = global_weights
            for _ in range(edge_rounds):
                edge_w, n_used = edge_round(edge_w, members, train_set, client_indices, rng, frac, local_epochs, lr, batch_size, device)
                client_updates_total += n_used
            n_edge = sum(len(client_indices[c]) for c in members)  
            edge_results.append((edge_w, n_edge))
            edge_cloud_msgs += 1

        set_weights(global_model, weighted_average(edge_results))
        loss, acc = evaluate(global_model, test_loader, device)
        history.append({"round": t, "loss": loss, "acc": acc, "client_updates": client_updates_total, "edge_cloud_msgs": edge_cloud_msgs})
        print(f"Cloud round {t:02d} | loss {loss:.3f} | acc {acc:.3f}")

    return history