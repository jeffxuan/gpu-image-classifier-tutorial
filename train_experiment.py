import csv
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

train_data = datasets.FashionMNIST("./data", True, download=True, transform=transform)
test_data = datasets.FashionMNIST("./data", False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

model = CNN().to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

history = []
start_time = time.perf_counter()

for epoch in range(1, EPOCHS + 1):
    model.train()
    loss_total = 0.0
    correct = 0
    count = 0

    for images, labels in train_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        count += labels.size(0)

    model.eval()
    test_correct = 0
    test_count = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(DEVICE, non_blocking=True))
            test_correct += (outputs.argmax(1) == labels.to(DEVICE)).sum().item()
            test_count += labels.size(0)

    row = {
        "epoch": epoch,
        "loss": loss_total / len(train_loader),
        "train_accuracy": correct / count,
        "test_accuracy": test_correct / test_count,
    }
    history.append(row)
    print(
        f"Epoch {epoch}/{EPOCHS} | Loss: {row['loss']:.4f} | "
        f"Train accuracy: {row['train_accuracy']:.2%} | "
        f"Test accuracy: {row['test_accuracy']:.2%}"
    )

elapsed = time.perf_counter() - start_time

with open("history.csv", "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=history[0].keys())
    writer.writeheader()
    writer.writerows(history)

model.eval()
with open("predictions.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["index", "label", "prediction", "confidence"])
    with torch.no_grad():
        for index, (image, label) in enumerate(test_data):
            probabilities = torch.softmax(model(image.unsqueeze(0).to(DEVICE)), dim=1)[0]
            prediction = probabilities.argmax().item()
            writer.writerow([index, label, prediction, probabilities[prediction].item()])

torch.save(model.state_dict(), "checkpoints/fashion_mnist_cnn_experiment.pt")
print(f"Training time: {elapsed:.2f} seconds")
print("Saved: history.csv, predictions.csv, checkpoints/fashion_mnist_cnn_experiment.pt")
