import csv
from collections import Counter

import matplotlib.pyplot as plt


names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


epochs = []
losses = []
train_accuracy = []
test_accuracy = []

with open("history.csv", newline="") as file:
    for row in csv.DictReader(file):
        epochs.append(int(row["epoch"]))
        losses.append(float(row["loss"]))
        train_accuracy.append(float(row["train_accuracy"]) * 100)
        test_accuracy.append(float(row["test_accuracy"]) * 100)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(epochs, losses, marker="o")
axes[0].set_title("Training loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(alpha=0.3)

axes[1].plot(epochs, train_accuracy, marker="o", label="Train")
axes[1].plot(epochs, test_accuracy, marker="o", label="Test")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig("training_curves.png", dpi=160)
plt.close(fig)

matrix = Counter()
with open("predictions.csv", newline="") as file:
    for row in csv.DictReader(file):
        matrix[(int(row["label"]), int(row["prediction"]))] += 1

values = [
    [matrix[(true_label, predicted_label)] for predicted_label in range(10)]
    for true_label in range(10)
]

fig, ax = plt.subplots(figsize=(9, 8))
image = ax.imshow(values, cmap="Blues")
fig.colorbar(image, ax=ax)
ax.set_title("Fashion-MNIST confusion matrix")
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_xticks(range(10), names, rotation=45, ha="right")
ax.set_yticks(range(10), names)

for row in range(10):
    for column in range(10):
        color = "white" if values[row][column] > max(max(line) for line in values) * 0.5 else "black"
        ax.text(column, row, values[row][column], ha="center", va="center", color=color)

fig.tight_layout()
fig.savefig("confusion_matrix.png", dpi=160)
print("Saved training_curves.png and confusion_matrix.png")
