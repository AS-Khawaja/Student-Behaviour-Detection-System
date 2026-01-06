# Student Behaviour Detection System

This repository contains an end-to-end **Student Behaviour Detection System** that predicts **student attention and emotional state** using facial images and structured metadata.  
The project combines **computer vision**, **deep learning**, and **multimodal feature fusion** to model student behaviour in classroom-like environments.

The system was developed as an academic project and focuses on **robust preprocessing**, **label aggregation**, and **transfer learning** using pre-trained CNNs.

---

## 📌 Key Features

- Multimodal learning using **facial images + metadata**
- Robust preprocessing of **CSV, JSON metadata, and image data**
- Majority-vote **label aggregation** from multiple human annotators
- Transfer learning using **ResNet-18, EfficientNet-B0, MobileNetV2**
- Hybrid **CNN + metadata neural network**
- Facemesh **PCA-based dimensionality reduction**
- Defensive handling of missing or noisy data
- Comprehensive evaluation using accuracy, precision, recall, F1-score, and confusion matrices

---

## 🧠 Problem Overview

Student engagement and emotional state are important indicators of learning effectiveness.  
This project aims to automatically detect:

- **Attention state** (e.g., attentive / inattentive)
- **Emotional state** (e.g., neutral, happy, sad, etc.)

using facial cues captured in images along with metadata such as:

- Head pose (pitch, yaw, roll)
- Face bounding box dimensions
- Estimated age and gender
- Facial landmarks (facemesh)
- Emotion probabilities from metadata

---

## 🗂️ Project Structure

```
.
├── preprocessing.py
├── deliverable2_pipeline.py
├── images_master*.csv
├── metadata/
├── labelers_jsons/
├── preproc_visuals/
├── deliverable2_outputs/
├── requirements.txt
└── README.md
```

---

## ⚙️ Data Preprocessing

The preprocessing pipeline performs the following steps:

1. Loads image paths and metadata references from CSV files  
2. Resolves metadata JSON paths  
3. Extracts facial features  
4. Aggregates labels from **four human annotators** using **majority voting**
5. Computes inter-annotator agreement scores
6. Encodes categorical labels and normalizes numeric features
7. Generates exploratory visualizations

Output: a **cleaned and model-ready CSV file**

---

## 🏗️ Model Architecture

### Image Backbone
- ResNet-18  
- EfficientNet-B0  
- MobileNetV2  

### Metadata Branch
- Fully connected neural network (MLP)

### Hybrid Fusion Model
- Concatenation of image embeddings and metadata features

---

## 🧪 Training & Evaluation

- Framework: **PyTorch**
- Optimizer: Adam
- Loss: Cross-Entropy Loss
- Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix

---

## 🚀 How to Run

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Preprocessing
```bash
python preprocessing.py
```

### Train the Model
```bash
python deliverable2_pipeline.py
```

---

## 📄 Disclaimer

This project is intended for **academic and research purposes only**.

---

## 👤 Author

**AS Khawaja**  
Computer Science Undergraduate  
Interests: Machine Learning, Computer Vision, Multimodal Systems
