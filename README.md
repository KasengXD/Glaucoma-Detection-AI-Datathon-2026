# 👁️ Glaucoma Detection AI: Datathon 2026
### Automated Screening & Explainable Diagnosis of Glaucomatous Optic Neuropathy (GON)

---

## 📜 Abstract
Glaucoma is often referred to as the *“Silent Thief of Sight”* due to its asymptomatic progression in early stages. This project presents a robust and efficient deep learning pipeline for automated glaucoma detection using fundus images.

The model leverages **EfficientNet-B0** for high-performance image classification and integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)** to enhance interpretability. This approach transforms the system from a "black-box" model into a clinically meaningful tool by highlighting regions of interest in the optic disc.

---

## 🔬 Scientific Methodology

### 1️⃣ Data Engineering
The system utilizes the **Hillel-Yaffe Glaucoma Dataset (HYGD)** with careful preprocessing to ensure clinical reliability:

- **Patient-Safe Splitting**  
  Implemented using `GroupShuffleSplit` to prevent data leakage by ensuring that images from the same patient do not appear in both training and validation sets.

- **Image Preprocessing**  
  - Resized to **224 × 224**
  - Normalized using **ImageNet statistics**
  - Optimized for transfer learning performance

---

### 2️⃣ Model Architecture & Explainability

- **EfficientNet-B0**  
  Chosen for its strong balance between:
  - Accuracy  
  - Computational efficiency  
  - Scalability  

- **Grad-CAM (XAI Integration)**  
  Provides visual explanations by:
  - Highlighting **optic nerve head regions**
  - Mapping important features influencing predictions  
  - Improving trust and usability in clinical settings  

---

## 📈 Quantitative Results

### 🔹 Performance Summary
The model achieves **excellent diagnostic performance**, particularly in detecting glaucoma cases (high sensitivity).

- **AUC Score:** 0.9902  
  → Indicates near-perfect class separability  

---

### 🔹 Confusion Matrix

|                      | Predicted Normal | Predicted Glaucoma |
|----------------------|----------------|--------------------|
| **Actual Normal**     | 31             | 2                  |
| **Actual Glaucoma**   | 2              | 97                 |

---

### 📊 Key Metrics
- **True Positives (TP):** 97  
- **True Negatives (TN):** 31  
- **False Positives (FP):** 2  
- **False Negatives (FN):** 2  

---

### 🧠 Performance Insights
- **High Sensitivity:** Critical for medical screening (minimizes missed glaucoma cases)  
- **Low False Negatives:** Only 2 misclassified glaucoma cases  
- **Low False Positives:** Reduces unnecessary concern for healthy patients  
- **Balanced Model:** Strong performance across both classes  

---

### 🧾 Conclusion
The model demonstrates **high reliability and clinical relevance** in glaucoma detection.  
With near-perfect AUC and minimal classification errors, it is well-suited for decision-support systems in ophthalmology.

---

## 🛠️ Project Implementation
This project is implemented using a modular deep learning pipeline designed for scalability, reproducibility, and real-time inference.

### 📂 Repository Structure
├── training.py              # Data Preprocessing & Model Training (EfficientNet-B0)
├── evaluate.py              # Performance Metrics (ROC, Confusion Matrix)
├── app.py                   # Streamlit Web Interface (Main Dashboard)
├── utils.py                 # Custom Dataset (HYGDDataset) & Image Transforms
├── Labels.csv               # Metadata & Clinical Labels (Hillel-Yaffe Dataset)
├── Images/                  # Dataset (Retinal Fundus Images)
├── results/                 # Folder for model artifacts and plots
│   ├── evaluation_metrics.png # Generated ROC (AUC 0.9902) & Confusion Matrix
│   └── training_history.png   # Loss and Accuracy curves over 10 epochs
├── LICENSE                  # MIT License
└── README.md                # Project Documentation & Guide

---

### ⚙️ Core Pipeline Workflow

1️⃣ **Data Loading**
- Custom dataset class (`HYGDDataset`) loads images and labels
- Applies preprocessing and transformations

2️⃣ **Preprocessing**
- Resize images to **224 × 224**
- Normalize using **ImageNet mean & std**
- Data prepared for transfer learning

3️⃣ **Model Training**
- Backbone: **EfficientNet-B0**
- Transfer learning with pretrained weights
- Fine-tuned on glaucoma dataset

4️⃣ **Evaluation**
- Performance measured using:
  - ROC Curve (AUC)
  - Confusion Matrix
- Metrics computed for clinical reliability

5️⃣ **Explainability (XAI)**
- Grad-CAM applied on trained model
- Highlights important regions in fundus images
- Improves interpretability for medical use

6️⃣ **Deployment / Inference**
- `app.py` provides real-time prediction
- Displays:
  - Predicted class (Normal / Glaucoma)
  - Confidence score
  - Grad-CAM heatmap overlay

---

### 🧪 Training Configuration

- **Framework:** PyTorch  
- **Model:** EfficientNet-B0  
- **Optimizer:** Adam  
- **Learning Rate:** 1e-4  
- **Loss Function:** CrossEntropyLoss  
- **Epochs:** 10  
- **Input Size:** 224 × 224  

---

### 💡 Design Considerations

- **Efficiency:** Lightweight architecture for fast inference  
- **Accuracy:** High AUC for reliable screening  
- **Explainability:** Grad-CAM for clinical trust  
- **Scalability:** Modular code structure for easy extension  

---
