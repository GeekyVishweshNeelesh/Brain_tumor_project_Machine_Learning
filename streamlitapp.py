import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.express as px
import os

# ===============================
# Page Config and Styles
# ===============================
st.set_page_config(page_title="Brain MRI Tumor Detection", layout="wide")
st.markdown(
    """
    <style>
        body, .stMarkdown, .stText, .stSubheader, .stRadio label, .stFileUploader label, .stButton button {
            font-size: 22px !important;
        }
        h1 { font-size: 32px !important; }
        h2, h3, h4, h5, h6 { font-size: 22px !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# ===============================
# TFLite Model Files
# ===============================
model_files = {
    "Custom CNN": "custom_cnn_model.tflite",
    "MobileNet": "mobilenet_model.tflite",
    "ResNet50": "resnet50_model.tflite"
}

# Debug log
st.write("**Available Model Files:**")
for k, v in model_files.items():
    st.write(f"- {k}: {os.path.exists(v)} ({v})")

# ===============================
# Load TFLite Model
# ===============================
def load_tflite_model(model_path):
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"❌ Failed to load model {model_path}: {e}")
        return None

# ===============================
# Image Preprocessing for TFLite
# ===============================
def preprocess_image(img, input_shape):
    img_resized = img.resize((input_shape[1], input_shape[2]))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ===============================
# Prediction with TFLite Model
# ===============================
def predict_tflite(interpreter, img_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    return prediction

# ===============================
# Streamlit UI
# ===============================
st.title("🧠 Brain MRI Tumor Detection")
st.markdown("Upload a brain MRI image and choose a model for prediction.")

model_choice = st.radio("**Select Model:**", list(model_files.keys()), horizontal=True)
uploaded_file = st.file_uploader("**Upload MRI Image:**", type=["jpg", "jpeg", "png"])
class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')

    # Load selected TFLite model
    interpreter = load_tflite_model(model_files[model_choice])
    if interpreter is not None:
        input_shape = interpreter.get_input_details()[0]['shape']  # e.g., (1, 224, 224, 3)
        img_array = preprocess_image(img, input_shape)

        # Make prediction
        prediction = predict_tflite(interpreter, img_array)
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Confidence Graph
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

        # Two Columns: Image left, Graph right
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
        st.error("Model failed to load. Please check the TFLite file.")
else:
    st.info("Please upload an MRI image to start prediction.")
