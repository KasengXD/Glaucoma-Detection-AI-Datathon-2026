import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GroupShuffleSplit
import os

# Import your custom dataset logic
from utils import HYGDDataset, get_transforms
from sklearn.metrics import precision_recall_curve, average_precision_score

def save_extra_metrics(all_labels, all_probs):
    plt.figure(figsize=(12, 5))
    
    # --- PRECISION-RECALL CURVE ---
    plt.subplot(1, 2, 1)
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    ap = average_precision_score(all_labels, all_probs)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (Positive Predictive Value)')
    plt.title(f'Precision-Recall Curve: AP={ap:.3f}')
    
    # --- PROBABILITY HISTOGRAM ---
    plt.subplot(1, 2, 2)
    plt.hist([p for l, p in zip(all_labels, all_probs) if l == 0], bins=20, alpha=0.5, label='Normal', color='blue')
    plt.hist([p for l, p in zip(all_labels, all_probs) if l == 1], bins=20, alpha=0.5, label='Glaucoma', color='red')
    plt.xlabel('Probability of Glaucoma')
    plt.ylabel('Number of Images')
    plt.title('Confidence Distribution')
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/extra_metrics.png')
    plt.show()

# --- SETTINGS ---
BASE_PATH = r"D:\hygd"
DATA_DIR = os.path.join(BASE_PATH, "Images")
CSV_PATH = os.path.join(BASE_PATH, "Labels.csv")
MODEL_PATH = "glaucoma_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    # 1. Load Data (Must match the split used in training)
    df = pd.read_csv(CSV_PATH).dropna(axis=1, how='all')
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, val_idx = next(gss.split(df, groups=df['Patient']))
    
    _, val_tf = get_transforms()
    val_ds = HYGDDataset(df.iloc[val_idx], DATA_DIR, val_tf)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    # 2. Load the Trained "Brain"
    model = models.resnet50() # Change to mobilenet_v2 if that's what you trained
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []

    # 3. Collect Predictions
    print("🧐 Evaluating model performance...")
    with torch.no_grad():
        for imgs, lbls, _ in val_loader:
            outputs = model(imgs.to(DEVICE))
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            all_probs.extend(probs[:, 1].cpu().numpy()) # Probability of Glaucoma
            all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
            all_labels.extend(lbls.numpy())

    # 4. Generate the Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color='royalblue', lw=3, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax1.fill_between(fpr, tpr, alpha=0.15, color='royalblue')
    ax1.plot([0, 1], [0, 1], color='slategray', lw=1.5, linestyle='--')
    ax1.set_title('Receiver Operating Characteristic (ROC)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('False Positive Rate (1 - Specificity)')
    ax1.set_ylabel('True Positive Rate (Sensitivity)')
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.2, linestyle='--')

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal (GON-)', 'Glaucoma (GON+)'])
    disp.plot(ax=ax2, cmap='Blues', values_format='d')
    ax2.set_title('Confusion Matrix - Glaucoma Detection')

    # 5. Save and Show
    os.makedirs('results', exist_ok=True)
    plt.tight_layout()
    plt.savefig('results/evaluation_metrics.png')
    print("✅ Results saved to results/evaluation_metrics.png")
    plt.show()

if __name__ == "__main__":
    evaluate()