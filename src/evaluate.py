import os
import csv
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.dataset import get_dataloaders
from src.model import build_model



data_dir = "data/pokemon"
batch_size = 32
device = "cuda" if torch.cuda.is_available() else "cpu"

model_paths = [
    "results/resnet18_freeze.pth",
    "results/resnet18_finetune.pth",
    "results/mobilenet_freeze.pth",
    "results/mobilenet_finetune.pth",
]



_, _, test_loader, classes = get_dataloaders(
    data_dir,
    batch_size=batch_size
)



def evaluate_model(model, test_loader):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return accuracy, precision, recall, f1



results = []

for path in model_paths:
    if not os.path.exists(path): 
        continue

    checkpoint = torch.load(path, map_location=device)

    experiment_name = checkpoint.get(
        "experiment_name",
        os.path.basename(path).replace(".pth", "")
    )

    if "mobilenet" in experiment_name:
        model_name = checkpoint.get("model_name", "mobilenet_v3")
    else:
        model_name = checkpoint.get("model_name", "resnet18")

    if "finetune" in experiment_name:
        freeze_backbone = checkpoint.get("freeze_backbone", False)
    else:
        freeze_backbone = checkpoint.get("freeze_backbone", True)

    model = build_model(
        model_name=model_name,
        num_classes=len(classes),
        pretrained=False,
        freeze_backbone=freeze_backbone
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    accuracy, precision, recall, f1 = evaluate_model(model, test_loader)

    results.append({
        "experiment": experiment_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    print(
        f"{experiment_name} | "
        f"Accuracy: {accuracy:.4f}, "
        f"Precision: {precision:.4f}, "
        f"Recall: {recall:.4f}, "
        f"F1: {f1:.4f}"
    )



os.makedirs("results", exist_ok=True)

df = pd.DataFrame(results)
df.to_csv("results/evaluation_results.csv", index=False)

print("\nEvaluation results saved to results/evaluation_results.csv")
print(df)
