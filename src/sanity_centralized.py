import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.model import SmallCNN
from src.training import evaluate, get_device, train_local

torch.manual_seed(0)
device = get_device()
print("Device :", device)

tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
train_set = datasets.FashionMNIST("data", train=True, download=True, transform=tfm)
test_set = datasets.FashionMNIST("data", train=False, download=True, transform=tfm)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=256)

model = SmallCNN()
for epoch in range(1, 4):
    loss = train_local(model, train_loader, epochs=1, lr=0.05, device=device)
    test_loss, acc = evaluate(model, test_loader, device)
    print(f"Epoch {epoch} | train loss {loss:.3f} | test acc {acc:.3f}")