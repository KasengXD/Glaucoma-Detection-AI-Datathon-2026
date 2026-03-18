👁️ Glaucoma Detection AI: Datathon 2026
Automated Screening & Explainable Diagnosis of Glaucomatous Optic Neuropathy (GON)

📜 Abstract
Early-stage glaucoma is often asymptomatic, earning it the title "The Silent Thief of Sight." This project presents a robust, efficient Deep Learning pipeline using EfficientNet-B0 for high-speed screening. By integrating Grad-CAM, we bridge the gap between "Black-Box" AI and clinical practice, providing doctors with heatmaps that highlight morphological changes in the retinal fundus.

🔬 Scientific Methodology
1. Data Engineering
We utilized the Hillel-Yaffe Glaucoma Dataset (HYGD). To ensure clinical validity:

Patient-Safe Splitting: Used GroupShuffleSplit to prevent data leakage—ensuring that the model is tested on patients it has never encountered during training.

Preprocessing: Images were standardized to 224x224 and normalized using ImageNet statistics to leverage transfer learning effectively.

2. Architecture & XAI
EfficientNet-B0: Selected for its state-of-the-art scaling and accuracy, providing a more powerful feature extractor than standard legacy models.

Grad-CAM Logic: We utilize the gradients of the final convolutional layer (model.features[-1]) to map the "regions of interest," ensuring the model focuses on the Optic Nerve Head rather than extraneous artifacts.

📈 Quantitative Results
Performance Summary
The model demonstrates exceptional reliability and high sensitivity, which is the primary requirement for a medical screening tool to avoid False Negatives.

AUC Score: 0.9902 (Demonstrating near-perfect separability).

Sensitivity: Correctly identified 97 out of 99 positive Glaucoma (GON+) cases.

Specificity: Successfully identified 31 out of 33 Normal (GON-) cases.

🛠️ Project Implementation
Repository Organization
Plaintext
├── app.py                # Dashboard for real-time inference
├── training.py           # Core training & optimization script
├── evaluate.py           # Advanced metrics (ROC, Confusion Matrix)
├── utils.py              # Custom Dataset (HYGDDataset) and Transforms
├── Labels.csv            # Ground truth annotations
├── Images/               # (Local only) Dataset directory
└── results/              # Performance visualizations (ROC, Confusion Matrix)

Installation & Deployment

Clone & Install:
pip install torch torchvision streamlit opencv-python pandas scikit-learn matplotlib plotly grad-cam

Train & Evaluate:
python training.py
python evaluate.py

Launch Clinical Dashboard:
python -m streamlit run app.py
🌟 Clinical Impact & Future Work
High-Precision Screening: With an AUC of 0.99, the tool is highly reliable for preliminary screenings.

Explainability: Grad-CAM visualizations help build trust with medical practitioners by justifying AI decisions.

Author: WONG KA SENG

Event: Datathon 2026

License: MIT
