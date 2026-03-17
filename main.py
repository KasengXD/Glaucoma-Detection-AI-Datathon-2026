import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet50_Weights
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
from utils import HYGDDataset, get_transforms, plot_bootcamp_results
import glob

# --- CONFIGURATION (Fixed for your D:\hygd structure) ---
csv_search = glob.glob(r"D:\hygd/**/Labels.csv", recursive=True)

if not csv_search:
    # If it's not named Labels.csv, maybe it's just in the main folder?
    if os.path.exists(r"D:\hygd\Labels.csv"):
        CSV_PATH = r"D:\hygd\Labels.csv"
    else:
        print("❌ ERROR: Could not find Labels.csv! Please check your folder.")
        exit()
else:
    CSV_PATH = csv_search[0]

# Automatically set the image folder based on where the CSV is
BASE_PATH = os.path.dirname(CSV_PATH)
# If Images is a folder next to the CSV, this finds it:
DATA_DIR = os.path.join(os.path.dirname(BASE_PATH), "Images") 

# If that fails, try the most likely direct path
if not os.path.exists(DATA_DIR):
    DATA_DIR = r"D:\hygd\Images"

print(f"✅ Success! Found Labels at: {CSV_PATH}")
print(f"✅ Success! Found Images at: {DATA_DIR}")

BATCH_SIZE = 8 
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def main():
    # Verify files exist before starting
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: Could not find {CSV_PATH}")
        return
    
    print(f"✅ Loading dataset from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH).dropna(axis=1, how='all')
    
    # SERIES 5: Patient-Safe Split
    # Ensures images from the same patient aren't in both Train and Test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df['Patient']))
    
    train_tf, val_tf = get_transforms()
    train_ds = HYGDDataset(df.iloc[train_idx], DATA_DIR, train_tf)
    val_ds = HYGDDataset(df.iloc[val_idx], DATA_DIR, val_tf)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # SERIES 6: CNN Training (ResNet50)
    print(f"🚀 Initializing Model on {DEVICE}...")
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, 2) 
    model = model.to(DEVICE)

    # Handle class imbalance (GON+ vs GON-)
    y_train = [1 if x == 'GON+' else 0 for x in df.iloc[train_idx]['Label']]
    cw = torch.FloatTensor(compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)).to(DEVICE)
    
    # Loss function + Quality Score weighting
    criterion = nn.CrossEntropyLoss(weight=cw, reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

    history = {'loss': [], 'acc': []}

    print("--- Starting Training Loop ---")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for imgs, lbls, wts in train_loader:
            imgs, lbls, wts = imgs.to(DEVICE), lbls.to(DEVICE), wts.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            
            # Apply image quality weights to the loss
            loss = (criterion(outputs, lbls) * wts).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # SERIES 7: Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, lbls, _ in val_loader:
                outputs = model(imgs.to(DEVICE))
                all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
                all_labels.extend(lbls.numpy())

        acc = accuracy_score(all_labels, all_preds)
        history['loss'].append(total_loss / len(train_loader))
        history['acc'].append(acc)
        scheduler.step(acc)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {history['loss'][-1]:.4f} | Val Acc: {acc:.4f}")

    # SERIES 8: Results Visualization
    plot_bootcamp_results(history)
    print("\n--- Final Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=['GON-', 'GON+']))
    
    # Save the 'Brain' for Series 9 (The App)
    torch.save(model.state_dict(), "glaucoma_model.pth")
    print("✅ Model Trained and Saved as 'glaucoma_model.pth'!")

if __name__ == "__main__":
    main()