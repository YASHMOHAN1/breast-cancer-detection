import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Breast Cancer Detection AI",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #0f172a;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
h1, h2, h3, p, label {
    color: white !important;
}
.stButton>button {
    width: 100%;
    border-radius: 12px;
    height: 3em;
    font-size: 18px;
}
.metric-card {
    background: #1e293b;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(BASE_DIR, "models", "best_model.pth")

    model = models.resnet18(weights=None)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    return model

model = load_model()

# -----------------------------
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

classes = ["Benign", "Malignant"]

# -----------------------------
# HEADER
# -----------------------------
st.markdown("# 🩺 Breast Cancer Detection AI")
st.markdown("### Deep Learning Detection using Histopathological Images")

st.divider()

# -----------------------------
# LAYOUT
# -----------------------------
col1, col2 = st.columns([1,1])

# -----------------------------
# LEFT SIDE
# -----------------------------
with col1:
    st.subheader("📤 Upload Image")

    uploaded_file = st.file_uploader(
        "Choose Histopathology Image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

# -----------------------------
# RIGHT SIDE
# -----------------------------
with col2:
    st.subheader("📊 Prediction Result")

    if uploaded_file is not None:

        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        label = classes[predicted.item()]
        conf = confidence.item() * 100

        if label == "Benign":
            st.success(f"✅ Prediction: {label}")
        else:
            st.error(f"⚠️ Prediction: {label}")

        st.info(f"Confidence Score: {conf:.2f}%")

        benign_prob = probs[0][0].item() * 100
        malignant_prob = probs[0][1].item() * 100

        st.subheader("Probability Distribution")
        st.progress(int(max(benign_prob, malignant_prob)))

        st.write(f"Benign: {benign_prob:.2f}%")
        st.write(f"Malignant: {malignant_prob:.2f}%")

# -----------------------------
# FOOTER
# -----------------------------
st.divider()
st.caption("Developed using PyTorch + Streamlit + ResNet18")
