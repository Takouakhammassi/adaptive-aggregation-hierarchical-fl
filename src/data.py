import numpy as np
from torchvision import datasets, transforms

NORMALIZE = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
)


def load_fashion_mnist(root: str = "data"):
    train_set = datasets.FashionMNIST(root, train=True, download=True, transform=NORMALIZE)
    test_set = datasets.FashionMNIST(root, train=False, download=True, transform=NORMALIZE)
    return train_set, test_set


def dirichlet_partition(labels, num_clients: int, alpha: float, seed: int = 0, min_size: int = 10):
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    num_classes = int(labels.max()) + 1

    while True:  
        client_indices = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            idx_c = np.where(labels == c)[0]
            rng.shuffle(idx_c)
            proportions = rng.dirichlet(alpha * np.ones(num_clients))
            cuts = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
            for client, part in enumerate(np.split(idx_c, cuts)):
                client_indices[client].extend(part.tolist())
        if min(len(ci) for ci in client_indices) >= min_size:
            break

    return [np.array(ci) for ci in client_indices]


def class_distribution(labels, client_indices, num_classes: int = 10):
    labels = np.asarray(labels)
    mat = np.zeros((len(client_indices), num_classes), dtype=int)
    for i, idx in enumerate(client_indices):
        mat[i] = np.bincount(labels[idx], minlength=num_classes)
    return mat