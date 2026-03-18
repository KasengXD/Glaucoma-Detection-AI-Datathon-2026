import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import os

# Import Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Glaucoma AI Dashboard", page_icon="👁️", layout="wide")

# --- SETTINGS ---
MODEL_PATH = "efficientnet_glaucoma.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['Normal (GON-)', 'Glaucoma (GON+)']


# --- CACHE THE MODEL ---
@st.cache_resource
def load_trained_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        st.warning(f"⚠️ Model weights not found at '{MODEL_PATH}'. Please ensure you have trained the model.")

    model.to(DEVICE)
    model.eval()
    return model


# --- TRANSFORMS ---
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# --- 1. YOUR CUSTOM THRESHOLD LOGIC ---
def interpret_probability(prob):
    """Maps the raw probability to a specific confidence category."""
    if prob >= 0.95:
        return "Very confident glaucoma"
    elif prob >= 0.70:
        return "Moderate confidence"
    elif prob >= 0.52:
        return "Uncertain prediction"
    else:
        return "Low probability of glaucoma"


# --- 2. NEW DIAGNOSTIC SUMMARY GENERATOR ---
def get_assessment_summary(probs):
    """Generates a dynamic text summary based on the interpreted probability."""
    glaucoma_prob = probs[1]  # We specifically look at the Glaucoma probability score
    interpretation = interpret_probability(glaucoma_prob)

    if interpretation == "Very confident glaucoma":
        text = f"The AI strongly indicates the presence of Glaucoma (GON+) with **{glaucoma_prob * 100:.1f}% probability**. Please review the Grad-CAM heatmap above to inspect the specific regions (e.g., optic disc cupping, nerve fiber layer defects) that triggered this assessment."
    elif interpretation == "Moderate confidence":
        text = f"The AI detects potential signs of Glaucoma with **{glaucoma_prob * 100:.1f}% probability**. While indicative of GON+, expert correlation with the highlighted Grad-CAM areas is highly recommended."
    elif interpretation == "Uncertain prediction":
        text = f"The AI is uncertain (**{glaucoma_prob * 100:.1f}% probability** for Glaucoma). This scan likely contains ambiguous or overlapping features. Specialist review is strongly advised."
    else:
        # Low probability of glaucoma means it's likely Normal
        text = f"The AI predicts a **low probability ({glaucoma_prob * 100:.1f}%)** of Glaucoma. The model did not detect significant structural anomalies typically associated with the disease."

    return interpretation, text


# --- INFERENCE FUNCTION ---
def analyze_image(image, model):
    raw_image = image.convert('RGB').resize((224, 224))
    raw_image_np = np.array(raw_image) / 255.0

    transform = get_transforms()
    tensor_img = transform(raw_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor_img)
        probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]

    pred_idx = np.argmax(probs)
    prediction = CLASSES[pred_idx]

    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=tensor_img, targets=None)[0, :]
    cam_overlay = show_cam_on_image(raw_image_np, grayscale_cam, use_rgb=True)

    return raw_image, cam_overlay, probs, prediction


# --- DASHBOARD UI ---
def main():
    st.title("👁️ AI Glaucoma Diagnostic Dashboard")
    st.markdown("Upload a retinal fundus image to receive an instant AI assessment and Grad-CAM visual explanation.")

    model = load_trained_model()

    with st.sidebar:
        st.header("Patient Data")
        uploaded_file = st.file_uploader("Upload Retinal Scan (JPG/PNG)", type=['jpg', 'jpeg', 'png'])
        st.info("The AI uses EfficientNet-B0 and Grad-CAM to localize diagnostic features.")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        with st.spinner('Analyzing scan...'):
            raw_img, cam_overlay, probs, prediction = analyze_image(image, model)

        # 1. Display Results Header
        if prediction == 'Normal (GON-)':
            st.success(f"### Predicted Class: {prediction}")
        else:
            st.error(f"### Predicted Class: {prediction}")

        st.divider()

        # 2. Display 3-Panel Layout
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Original Scan")
            st.image(raw_img, use_container_width=True)

        with col2:
            st.subheader("Grad-CAM Explanation")
            st.image(cam_overlay, use_container_width=True, caption="Heatmap of AI attention")

        with col3:
            st.subheader("Confidence Scores")
            colors = ['#1f77b4' if p < 0.52 else '#d62728' for p in
                      probs]  # Updated color threshold to match your 0.52 logic
            fig = go.Figure(go.Bar(
                x=probs,
                y=CLASSES,
                orientation='h',
                marker_color=colors,
                text=[f"{p:.1%}" for p in probs],
                textposition='auto'
            ))
            fig.update_layout(
                xaxis=dict(range=[0, 1], tickformat=".0%"),
                margin=dict(l=0, r=0, t=30, b=0),
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

        # 3. Display the Updated Summary
        st.divider()
        st.subheader("📝 Diagnostic Summary")
        interpretation, summary_text = get_assessment_summary(probs)

        # Color code the summary box based on the specific interpretation
        if interpretation == "Very confident glaucoma":
            st.error(f"**AI Assessment:** {interpretation}\n\n{summary_text}")
        elif interpretation == "Moderate confidence":
            st.warning(f"**AI Assessment:** {interpretation}\n\n{summary_text}")
        elif interpretation == "Uncertain prediction":
            st.info(f"**AI Assessment:** {interpretation}\n\n{summary_text}")
        else:
            st.success(f"**AI Assessment:** {interpretation}\n\n{summary_text}")

    else:
        st.info("👈 Please upload an image from the sidebar to begin.")


if __name__ == "__main__":
    main()
