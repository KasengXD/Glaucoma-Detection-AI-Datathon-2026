👁️ Glaucoma Detection AI: Datathon 2026
Automated Screening & Explainable Diagnosis of Glaucomatous Optic Neuropathy (GON)

📜 Abstract
Early-stage glaucoma is often asymptomatic, earning it the title "The Silent Thief of Sight." This project presents a robust, efficient Deep Learning pipeline using EfficientNet-B0 for high-speed screening. By integrating Grad-CAM, we bridge the gap between "Black-Box" AI and clinical practice, providing doctors with heatmaps that highlight morphological changes in the optic disc.

🔬 Scientific Methodology
1. Data Engineering
The system utilizes the Hillel-Yaffe Glaucoma Dataset (HYGD). To ensure clinical validity:

Patient-Safe Splitting: Utilizes GroupShuffleSplit to prevent data leakage, ensuring the model is validated on patients it has never encountered during training.

Preprocessing: Images are standardized to 224x224 and normalized using standard ImageNet statistics to leverage transfer learning effectively.

2. Architecture & XAI
EfficientNet-B0: Selected as the core architecture for its state-of-the-art balance between parameter efficiency and accuracy.

Grad-CAM Logic: The pipeline utilizes the gradients of the final convolutional layer to map "regions of interest," ensuring the diagnostic focus remains on the Optic Nerve Head.

📈 Quantitative Results
Performance Summary
The model demonstrates exceptional reliability and high sensitivity, which is the primary requirement for a medical screening tool to avoid False Negatives.

AUC Score: 0.9902 (Demonstrating near-perfect separability).

Confusion Matrix: Correctly identified 97 out of 99 positive Glaucoma (GON+) cases.

Reliability: Achieved a high True Positive Rate while maintaining low False Positives (31 out of 33 Normal cases correctly identified).

🛠️ Project Implementation
Repository Organization
Plaintext
├── app.py                # Dashboard for real-time inference and Grad-CAM visualization
├── training.py           # Core training script using EfficientNet-B0
├── utils.py              # Custom HYGDDataset logic and image transforms
├── Labels.csv            # Clinical ground truth annotations
├── Images/               # Local dataset directory
└── results/              # Saved model weights and performance visualizations
Technical Specifications
Framework: PyTorch

Optimizer: Adam with a learning rate of 1e-4

Loss Function: CrossEntropyLoss

Training Duration: 10 Epochs
