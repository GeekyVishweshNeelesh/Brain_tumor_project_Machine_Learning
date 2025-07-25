
# 🧠 Brain Tumor Classification – Deep Learning Project

## **📖 Overview**
This project implements a **deep learning-based brain tumor classification system** using MRI images. It classifies brain MRI scans into **four categories: glioma, meningioma, pituitary tumor, and no tumor**.  
We built both a **custom CNN model** and **transfer learning models** (MobileNetV2 and ResNet50), compared their performance, and deployed the best-performing model using **Streamlit** for real-time predictions.

---

## **🎯 Objectives**
- Analyze brain MRI images and identify tumor types.
- Build and compare **Custom CNN** vs **Transfer Learning models**.
- Deploy a **user-friendly web app** for real-time predictions.
- Evaluate models using **accuracy, precision, recall, and F1-score**.

---

## **📂 Dataset**
The dataset used is publicly available on Google Drive:  
[**Brain MRI Dataset – Tumor Classification**]
(https://drive.google.com/drive/folders/1RnlEPbrYkfLi-s5lCtsMIG-9sgWF_FFI)  

The dataset contains MRI images categorised into:  
- **Glioma**  
- **Meningioma**  
- **Pituitary Tumor**  
- **No Tumor**

---

## **📌 Project Workflow**
1. **Data Understanding & Exploration**  
   - Dataset structure, sample images, class distribution.  
2. **Data Preprocessing**  
   - Normalised pixel values (0–1), resized images to 224x224.  
3. **Data Augmentation**  
   - Rotations, flips, zoom, brightness adjustments for generalisation.  
4. **Model Building**  
   - **Custom CNN** with Conv2D, MaxPooling, BatchNorm, Dropout.  
   - **Transfer Learning** (MobileNetV2, ResNet50 fine-tuned).  
5. **Model Training**  
   - EarlyStopping and ModelCheckpoint used.  
6. **Model Evaluation**  
   - Accuracy, Precision, Recall, F1-score, Confusion Matrix.  
7. **Model Comparison & Deployment**  
   - Best model deployed on a **Streamlit dashboard**.

---

## **🗂 Folder Structure**
```
brain-tumor-classification/
│
├── README.md
├── requirements.txt
│
├── notebook/
│ └── brain_tumor_classification.ipynb
│
└── streamlit_app/
```
## **⚙️ Installation & Setup**

### **1. Clone the Repository**
```bash
git clone https://github.com/<your-username>/brain-tumor-classification.git
cd brain-tumor-classification
```

## 2. Install Dependencies
``` bash
pip install -r requirements.txt
```

## 3. Run the Streamlit App 
``` bash
cd streamlit_app
streamlit run app.py
```

🚀 Usage

- Upload an MRI image via the Streamlit interface.

- The app displays:
  
- Predicted Tumor Type (glioma, meningioma, pituitary, or no tumor).

- Model Confidence Score.

📊 Model Performance

- Custom CNN Accuracy: ~85–88%.

- MobileNetV2 Accuracy: ~90–92%.

- ResNet50 Accuracy (final model): ~93–94%.

- Metrics used: Accuracy, Precision, Recall, F1-score, and Confusion Matrix.

🔖 Tags

Deep Learning, Image Classification, Medical Imaging, Brain MRI Analysis, CNN, Transfer Learning, TensorFlow, Keras, PyTorch, Data Augmentation, Data Preprocessing, Model Evaluation, Performance Metrics, Streamlit Deployment, Confusion Matrix, Accuracy & Loss Visualisation, Model Comparison, Healthcare AI, Computer Vision, Deployment Ready Applications, AI in Radiology






