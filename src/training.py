import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_local(model, loader: DataLoader, epochs: int, lr: float, device) -> float:
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    last_loss = 0.0
    for _ in range(epochs):
        total, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
            n += len(y)
        last_loss = total / max(n, 1)
    return last_loss


@torch.no_grad()
def evaluate(model, loader: DataLoader, device):
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        total_loss += criterion(out, y).item()
        correct += (out.argmax(dim=1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n