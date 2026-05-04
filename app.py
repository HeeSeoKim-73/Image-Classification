import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from src.model import build_model


device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint_path = "results/resnet18_freeze.pth"

checkpoint = torch.load(checkpoint_path, map_location=device)
classes = checkpoint["classes"]

model = build_model(
    model_name="resnet18",
    num_classes=len(classes),
    pretrained=False,
    freeze_backbone=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

st.title("Pokemon Classifier")


uploaded_file = st.file_uploader(
    "Upload pokemon image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x)
        prob = torch.softmax(output, dim=1)

        k = min(5, len(classes))
        topk = torch.topk(prob, k)

    top1_idx = topk.indices[0][0].item()
    top1_class = classes[top1_idx]
    top1_score = topk.values[0][0].item()

    st.subheader(f"Top-{k} Predictions")

    for score, idx in zip(topk.values[0], topk.indices[0]):
        class_name = classes[idx.item()]
        confidence = score.item() * 100
        st.write(f"{class_name}: {confidence:.2f}%")

    st.markdown("---")
    st.success(f"It's {top1_class}!")
    