"""Local Gradio interface for the fine-tuned dog-breed classifier."""

import csv
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
TEMP_DIR = PROJECT_DIR / ".gradio_tmp"
TEMP_DIR.mkdir(exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", str(TEMP_DIR))

import gradio as gr
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet34

CHECKPOINT_PATH = PROJECT_DIR / "resnet34_finetune_best.pt"
CLASSES_PATH = PROJECT_DIR / "classes.csv"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_classes():
    with CLASSES_PATH.open(newline="") as file:
        rows = list(csv.reader(file))
    return [row[0] for row in rows if row]


classes = load_classes()
model = resnet34(weights=None)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, len(classes)),
)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.to(DEVICE).eval()

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def predict(image):
    if image is None:
        raise gr.Error("Please upload a dog image.")

    tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        scores, indices = torch.topk(probabilities, k=5)
    return {classes[index]: float(score) for score, index in zip(indices.cpu(), scores.cpu())}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Dog image"),
    outputs=gr.Label(num_top_classes=5, label="Top 5 predicted breeds"),
    title="Dog Breed Classifier",
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
