import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.dataset import get_dataloaders
from src.model import build_model


data_dir = "data/pokemon"
batch_size = 32
epochs = 10
learning_rate = 1e-4

device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("results", exist_ok=True)



experiments = [
    {
        "name": "resnet18_freeze",
        "model_name": "resnet18",
        "pretrained": True,
        "freeze_backbone": True
    },
    {
        "name": "resnet18_finetune",
        "model_name": "resnet18",
        "pretrained": True,
        "freeze_backbone": False
    },
    {
        "name": "mobilenet_freeze",
        "model_name": "mobilenet_v3",
        "pretrained": True,
        "freeze_backbone": True
    },
    {
        "name": "mobilenet_finetune",
        "model_name": "mobilenet_v3",
        "pretrained": True,
        "freeze_backbone": False
    }
]



train_loader, val_loader, test_loader, classes = get_dataloaders(
    data_dir,
    batch_size=batch_size
)


for exp in experiments:
    print("\n" + "=" * 50)
    print(f"Experiment: {exp['name']}")
    print("=" * 50)

    model = build_model(
        model_name=exp["model_name"],
        num_classes=len(classes),
        pretrained=exp["pretrained"],
        freeze_backbone=exp["freeze_backbone"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )

    loss_list = []
    acc_list = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(
            train_loader,
            desc=f"{exp['name']} Epoch {epoch + 1}/{epochs}"
        ):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        loss_list.append(train_loss)
        acc_list.append(train_acc)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.4f}"
        )

    plt.figure()
    plt.plot(loss_list)
    plt.title(f"{exp['name']} Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"results/{exp['name']}_loss.png")
    plt.close()

    plt.figure()
    plt.plot(acc_list)
    plt.title(f"{exp['name']} Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(f"results/{exp['name']}_acc.png")
    plt.close()

    print(f"Learning curve saved for {exp['name']}")
    
    save_path = f"results/{exp['name']}.pth"

    torch.save({
        "experiment_name": exp["name"],
        "model_name": exp["model_name"],
        "pretrained": exp["pretrained"],
        "freeze_backbone": exp["freeze_backbone"],
        "model_state_dict": model.state_dict(),
        "classes": classes
    }, save_path)

    print(f"Saved model: {save_path}")

print("\nAll experiments finished.")
