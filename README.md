
# **Brain Tumor Classification using Deep Learning**

This project implements a **deep learning-based brain tumor classification system** using MRI images. The goal is to classify MRI scans into **four categories: glioma, meningioma, pituitary tumor, and no tumor**, using both a **custom CNN** and **transfer learning models** (MobileNetV2 and ResNet50).  

A **Streamlit web application** is included for real-time tumor predictions from uploaded MRI images.

---

## **📌 Project Workflow**
1. **Understand the Dataset** – Explore image distribution, check for class imbalance, and visualize sample images.  
2. **Data Preprocessing** – Normalize pixel values (0–1) and resize all images to 224x224 pixels.  
3. **Data Augmentation** – Apply transformations like rotation, flipping, zooming, and brightness adjustments to improve generalization.  
4. **Model Building**  
   - Custom CNN from scratch.  
   - Transfer Learning using MobileNetV2 & ResNet50 (ImageNet weights).  
5. **Model Training** – With callbacks like EarlyStopping and ModelCheckpoint.  
6. **Model Evaluation** – Accuracy, Precision, Recall, F1-score, Confusion Matrix.  
7. **Model Comparison** – Select the best-performing model.  
8. **Streamlit Deployment** – Web app for real-time classification.

---

## **📦 Project Deliverables**
- **Trained Models:**  
  - `custom_cnn_model.h5`  
  - `mobilenetv2_model.h5`  
  - `resnet50_model.h5`
- **Notebook:** `brain_tumor_classification.ipynb` (all steps: EDA, training, evaluation).
- **Streamlit App:** `streamlit_app/app.py`.
- **Public GitHub Repository** with clean, modular, and well-commented code.

---

## **📂 Folder Structure**
brain-tumor-classification/
│
├── README.md
├── requirements.txt
|
├── notebook/
│ └── brain_tumor_classification.ipynb
|
└── streamlit_app/
└── app.py


---

## **⚙️ Installation & Setup**

1. Clone the Repository

git clone https://github.com/<your-username>/brain-tumor-classification.git
cd brain-tumor-classification


2. Install Dependencies

pip install -r requirements.txt

3. Run the Streamlit App

cd streamlit_app
streamlit run app.py


🚀 Usage

    Upload an MRI image via the Streamlit interface.

    The app will display:

        Predicted Tumor Type (glioma, meningioma, pituitary, or no tumor).

        Model Confidence Score.



