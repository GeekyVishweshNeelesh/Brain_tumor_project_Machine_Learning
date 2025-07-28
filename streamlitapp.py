import os
import gdown
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import plotly.express as px

# ===============================
# Google Drive Model Links (Replace with Your IDs)
# ===============================
drive_links = {
    "Custom CNN": "https://drive.google.com/uc?id=1DJcA80r1TZZD1te5gi6boG93LqTN1aEK",
    "MobileNet": "https://drive.google.com/uc?id=1eykC2qsMiQ_aT6Hh7IjNDL2YcEitIwpB",
    "ResNet50": "https://drive.google.com/uc?id=1fr-zLQR94FOA_PrBpx3YhWkGUWq6-npc"
}

model_files = {
    "Custom CNN": "models/custom_cnn_model.h5",
    "MobileNet": "models/mobilenet_model.h5",
    "ResNet50": "models/resnet50_model.h5"
}

# Create models directory
os.makedirs("models", exist_ok=True)

# ===============================
# Download Model If Missing
# ===============================
def download_model_if_needed(model_name):
    model_path = model_files[model_name]
    if not os.path.exists(model_path):
        st.warning(f"Downloading {model_name} model... This might take a few seconds.")
        gdown.download(drive_links[model_name], model_path, quiet=False)
    return model_path

# ===============================
# Load Model
# ===============================
@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# ===============================
# Streamlit UI Config
# ===============================
st.set_page_config(page_title="Brain MRI Tumor Detection", layout="wide")
st.title("🧠 Brain MRI Tumor Detection")
st.markdown("Upload a brain MRI image and choose a model for prediction.")

# ===============================
# Model Selection
# ===============================
model_choice = st.radio("**Select Model:**", list(model_files.keys()), horizontal=True)
model_path = download_model_if_needed(model_choice)
model = load_model(model_path)

# ===============================
# File Uploader
# ===============================
uploaded_file = st.file_uploader("**Upload MRI Image:**", type=["jpg", "jpeg", "png"])
class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# ===============================
# Prediction and UI
# ===============================
if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')

    # Resize image according to model input shape
    input_shape = model.input_shape[1:3]
    img_resized = img.resize(input_shape)
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Confidence Scores
    confidence_scores = {class_labels[i]: float(prediction[0][i]) for i in range(len(class_labels))}
    fig = px.bar(
        x=list(confidence_scores.keys()),
        y=list(confidence_scores.values()),
        labels={'x': "Tumor Type", 'y': "Confidence Score"},
        text=[f"{v * 100:.1f}%" for v in confidence_scores.values()],
        color=list(confidence_scores.keys()),
        title="Confidence Scores"
    )
    fig.update_layout(
        xaxis=dict(title_font=dict(size=22), tickfont=dict(size=20)),
        yaxis=dict(title_font=dict(size=22), tickfont=dict(size=20)),
        title_font=dict(size=30),
        width=900,
        height=600,
        uniformtext_minsize=18,
        uniformtext_mode='hide'
    )

    # Layout: Image (left) and Graph (right)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(img, caption="Uploaded MRI Image", use_container_width=True)
    with col2:
        st.subheader("Prediction Results")
        st.markdown(f"**Model Used:** {model_choice}")
        st.markdown(f"**Predicted Tumor Type:** {predicted_class}")
        st.markdown(f"**Confidence:** {confidence:.2f}%")
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please upload an MRI image to start prediction.")
