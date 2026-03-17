import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.set_page_config(page_title="Glaucoma AI App")
st.title("👁️ Glaucoma Prediction Dashboard")

@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("glaucoma_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Upload a Fundus Image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Fundus Image", use_container_width=True)
    
    if st.button("Predict"):
        img_tensor = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.nn.functional.softmax(output[0], dim=0)
            pred = torch.argmax(prob).item()
        
        if pred == 1:
            st.error(f"Prediction: GON+ (Glaucoma) | Confidence: {prob[1]*100:.2f}%")
        else:
            st.success(f"Prediction: GON- (Healthy) | Confidence: {prob[0]*100:.2f}%")