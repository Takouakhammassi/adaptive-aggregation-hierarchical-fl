import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.model import SmallCNN
from src.training import evaluate, train_local


def get_weights(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def set_weights(model, weights):
    model.load_state_dict(weights)


def weighted_average(updates):
    total = sum(n for _, n in updates)
    avg = {}
    for key in updates[0][0].keys():
        avg[key] = sum(w[key].float() * (n / total) for w, n in updates)
    return avg


def client_update(global_weights, dataset, indices, epochs, lr, batch_size, device):
    model = SmallCNN()
    set_weights(model, global_weights)
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=batch_size, shuffle=True)
    train_local(model, loader, epochs=epochs, lr=lr, device=device)
    return get_weights(model), len(indices)


def run_fedavg(train_set, test_loader, client_indices, rounds=30, frac=0.3, local_epochs=1, lr=0.05, batch_size=32, seed=0, device="cpu"):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)  

    global_model = SmallCNN()
    num_clients = len(client_indices)
    m = max(1, int(frac * num_clients))
    history = []

    for r in range(1, rounds + 1):
        selected = rng.choice(num_clients, size=m, replace=False)
        global_weights = get_weights(global_model)

        updates = [
            client_update(global_weights, train_set, client_indices[c], local_epochs, lr, batch_size, device)
            for c in selected
        ]

        set_weights(global_model, weighted_average(updates))
        loss, acc = evaluate(global_model, test_loader, device)
        history.append({"round": r, "loss": loss, "acc": acc})
        print(f"Round {r:02d} | loss {loss:.3f} | acc {acc:.3f}")

    return history