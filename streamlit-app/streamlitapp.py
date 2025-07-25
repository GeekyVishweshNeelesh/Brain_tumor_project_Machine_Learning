import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import plotly.express as px
import os
import gdown

# ===============================
# Auto-download Models (if not exists)
# ===============================
MODEL_URLS = {
    "custom_cnn_model.h5": "https://drive.google.com/uc?id=1DJcA80r1TZZD1te5gi6boG93LqTN1aEK",
    "mobilenet_model.h5": "https://drive.google.com/uc?id=1eykC2qsMiQ_aT6Hh7IjNDL2YcEitIwpB",
    "resnet50_model.h5": "https://drive.google.com/uc?id=1fr-zLQR94FOA_PrBpx3YhWkGUWq6",
}

def download_models():
    for model_name, url in MODEL_URLS.items():
        if not os.path.exists(model_name):
            with st.spinner(f"Downloading {model_name}..."):
                gdown.download(url, model_name, quiet=False)

download_models()

# ===============================
# Custom CSS for Font Sizes
# ===============================
st.markdown(
    """
    <style>
        body, .stMarkdown, .stText, .stSubheader, .stRadio label, .stFileUploader label, .stButton button {
            font-size: 22px !important;
        }
        h1 {
            font-size: 32px !important;
        }
        h2, h3, h4, h5, h6 {
            font-size: 22px !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ===============================
# Model Files
# ===============================
model_files = {
    "Custom CNN": "custom_cnn_model.h5",
    "MobileNet": "mobilenet_model.h5",
    "ResNet50": "resnet50_model.h5"
}

# ===============================
# Load Model Dynamically
# ===============================
@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="Brain MRI Tumor Detection", layout="wide")
st.title("🧠 Brain MRI Tumor Detection")
st.markdown("Upload a brain MRI image and choose a model for prediction.")

# Model selection
model_choice = st.radio("**Select Model:**", list(model_files.keys()), horizontal=True)
model = load_model(model_files[model_choice])

# File uploader
uploaded_file = st.file_uploader("**Upload MRI Image:**", type=["jpg", "jpeg", "png"])
class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# ===============================
# Prediction Logic with Two Columns
# ===============================
if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')

    # Resize image according to model input
    input_shape = model.input_shape[1:3]
    img_resized = img.resize(input_shape)
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Model prediction
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Prepare confidence data
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

    # Create two columns: image on left, graph on right
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
