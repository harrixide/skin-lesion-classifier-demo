import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os

# -----------------------
# Path setup
# -----------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))     # app/
PROJECT_ROOT = os.path.dirname(BASE_DIR)                 # project root

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "src",
    "artifacts",
    "resnet18_skin.pt"
)

METADATA_PATH = os.path.join(
    PROJECT_ROOT,
    "src",
    "metadata.csv"
)

DEVICE = "cpu"

# -----------------------
# App config
# -----------------------

st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🩺",
    layout="centered"
)

# -----------------------
# Load metadata
# -----------------------

@st.cache_data
def load_metadata():
    return pd.read_csv(METADATA_PATH)

metadata_df = load_metadata()

LABEL_MAP = {
    "nv": "Benign",
    "bkl": "Benign",
    "df": "Benign",
    "vasc": "Benign",
    "mel": "Malignant",
    "bcc": "Malignant",
    "akiec": "Malignant"
}

# -----------------------
# Load model
# -----------------------

@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.eval()
    return model

model = load_model()

# -----------------------
# Image transforms
# -----------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------
# UI
# -----------------------

st.title("Skin Lesion Risk Classifier")

st.write(
    "Upload a skin lesion image to estimate whether it is **benign** or **malignant**."
)

st.caption(
    "Research demo only. This tool is not a medical diagnosis."
)

# -----------------------
# Supported classes
# -----------------------

st.subheader("Supported Lesion Types")

st.markdown("""
**Benign lesions**
- Melanocytic nevi (`nv`)
- Benign keratosis-like lesions (`bkl`)
- Dermatofibroma (`df`)
- Vascular lesions (`vasc`)

**Malignant lesions**
- Melanoma (`mel`)
- Basal cell carcinoma (`bcc`)
- Actinic keratoses / intraepithelial carcinoma (`akiec`)
""")

st.caption(
    "Ground-truth validation is available **only** for images originating "
    "from the HAM10000 dataset."
)

# -----------------------
# File upload
# -----------------------

uploaded_file = st.file_uploader(
    "Drag and drop an image here (JPG or PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)

    benign_prob = probs[0][0].item()
    malignant_prob = probs[0][1].item()

    if malignant_prob > 0.6:
        verdict = "Malignant"
    elif malignant_prob < 0.4:
        verdict = "Benign"
    else:
        verdict = "Uncertain"

    st.subheader("Prediction")
    st.write(f"**Verdict:** {verdict}")
    st.write(f"Benign probability: **{benign_prob * 100:.2f}%**")
    st.write(f"Malignant probability: **{malignant_prob * 100:.2f}%**")
    st.progress(malignant_prob)

    st.caption(
        "Low-confidence predictions are labeled as uncertain and should be "
        "interpreted with caution."
    )

    # -----------------------
    # Ground truth lookup
    # -----------------------

    filename = uploaded_file.name
    image_id = os.path.splitext(filename)[0]

    match = metadata_df[metadata_df["image_id"] == image_id]

    st.subheader("Ground Truth (Dataset Images Only)")

    if len(match) == 1:
        dx = match.iloc[0]["dx"]
        true_class = LABEL_MAP[dx]

        st.write(f"True diagnosis: **{true_class}**")

        if verdict == true_class:
            st.success("Model prediction is CORRECT for this image.")
        else:
            st.error("Model prediction is INCORRECT for this image.")
    else:
        st.info(
            "This image is not found in the dataset. "
            "Ground truth is unavailable."
        )
