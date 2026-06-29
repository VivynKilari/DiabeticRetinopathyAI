# 👁️ Explainable AI for Diabetic Retinopathy Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**An Explainable Artificial Intelligence System for Automated Diabetic Retinopathy Detection using Deep Learning**

</div>

---

# 📖 Overview

Diabetic Retinopathy (DR) is one of the leading causes of preventable blindness worldwide. Early diagnosis through retinal fundus imaging can significantly reduce the risk of vision loss.

This project develops an end-to-end AI-powered diagnostic system capable of:

- Detecting Diabetic Retinopathy severity
- Explaining predictions using Grad-CAM
- Generating clinical-style diagnostic reports
- Providing referral recommendations
- Deploying through a Streamlit web application

Instead of acting as a simple image classifier, this repository is designed to simulate a real-world AI-assisted clinical decision support system.

---

# ✨ Features

## Deep Learning

- EfficientNet-B0 Transfer Learning
- ImageNet Pretrained Backbone
- Weighted Sampling
- Medical Image Preprocessing
- Mixed Precision Training (Upcoming)

---

## Medical Image Processing

- Black Border Removal
- CLAHE Contrast Enhancement
- Image Normalization
- Retina Image Augmentation
- Albumentations Pipeline

---

## Explainable AI

- Grad-CAM Visualization
- Confidence Scores
- Lesion Highlighting
- Prediction Heatmaps

---

## Clinical Decision Support

- Disease Severity Classification (0–4)
- Referral Recommendation
- Clinical Report Generation
- Risk Assessment

---

## Deployment

- Streamlit Web Application
- Upload Retinal Images
- Interactive Predictions
- Download Clinical Reports

---

# 🩺 Disease Classes

| Grade | Diagnosis |
|---------|----------------------------|
| 0 | No Diabetic Retinopathy |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative Diabetic Retinopathy |

---

# 🗂 Dataset

Dataset Used:

**APTOS 2019 Blindness Detection**

Total Images:

- Training: 2930
- Validation: 366
- Testing: 366

Total:

3662 Retinal Fundus Images

---

# 🏗 Project Structure

```text
DiabeticRetinopathyAI/

│

├── datasets/

│

├── src/

│   ├── data/

│   │   ├── preprocessing.py

│   │   ├── dataset.py

│   │   ├── dataloader.py

│   │   ├── transforms.py

│   │   └── visualize.py

│

│   ├── models/

│   │   ├── model.py

│   │   └── test_model.py

│

│   ├── training/

│   │   ├── train.py

│   │   ├── engine.py

│   │   ├── losses.py

│   │   ├── metrics.py

│   │   ├── evaluate.py

│   │   └── utils.py

│

│   └── utils/

│       └── config.py

│

├── outputs/

│   └── models/

│

├── app/

│

├── requirements.txt

└── README.md
```

---

# ⚙️ Training Pipeline

```
Retinal Image

↓

Black Border Removal

↓

CLAHE Enhancement

↓

Albumentations

↓

Tensor Conversion

↓

EfficientNet-B0

↓

Prediction

↓

Grad-CAM

↓

Clinical Report
```

---

# 🧠 Model

Architecture:

- EfficientNet-B0
- Transfer Learning
- Dropout Layer
- Fully Connected Classification Layer
- 5 Output Classes

---

# 📊 Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Cohen's Kappa
- Confusion Matrix
- ROC Curve (Upcoming)

---

# 🖥 Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/DiabeticRetinopathyAI.git

cd DiabeticRetinopathyAI
```

Create Virtual Environment

```bash
python -m venv .venv
```

Activate

Windows

```bash
.venv\Scripts\activate
```

Linux / Mac

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Train the Model

```bash
python -m src.training.train
```

---

# 🔬 Test the Model

```bash
python -m src.models.test_model
```

---

# 🌐 Launch Web App

```bash
streamlit run app/app.py
```

---

# 📈 Future Improvements

- Vision Transformers
- EfficientNet-B3
- ConvNeXt
- Swin Transformer
- Focal Loss
- Mixed Precision Training
- TensorBoard Logging
- Early Stopping
- Clinical PDF Reports
- Multi-Dataset Training (EyePACS, IDRiD)
- Lesion Segmentation
- Docker Deployment

---

# 📚 Tech Stack

- Python
- PyTorch
- OpenCV
- Albumentations
- NumPy
- Pandas
- Matplotlib
- Streamlit
- scikit-learn
- timm

---

# 👨‍💻 Author

**Vivyn Kilari**

Robotics & Automation Engineering Student

Symbiosis Institute of Technology

---

# ⭐ Acknowledgements

- PyTorch
- timm
- OpenCV
- Albumentations
- APTOS 2019 Blindness Detection Dataset