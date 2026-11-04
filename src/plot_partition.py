import matplotlib.pyplot as plt

from src.data import class_distribution, dirichlet_partition, load_fashion_mnist

NUM_CLIENTS = 30
train_set, _ = load_fashion_mnist()
labels = train_set.targets.numpy()

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, alpha in zip(axes, [100, 0.5, 0.1]):
    parts = dirichlet_partition(labels, NUM_CLIENTS, alpha, seed=0)
    mat = class_distribution(labels, parts)
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_title(f"α = {alpha}")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Client")
    fig.colorbar(im, ax=ax, label="Nb d'exemples")

plt.suptitle("Répartition des classes par client (Dirichlet)")
plt.tight_layout()
plt.savefig("results/partition_dirichlet.png", dpi=150)
print("Figure sauvegardée dans results/partition_dirichlet.png")