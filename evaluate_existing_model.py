"""
Simple evaluation of existing hybrid model showing current performance
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import config
from dataset import get_data_loaders
from model import create_model

print("\n" + "="*70)
print("📊 EVALUATING EXISTING HYBRID MODEL")
print("="*70)

# Use existing best model
model_path = "models/hybrid_20251219_144329/best_model.pth"

if not os.path.exists(model_path):
    print(f"\n❌ Model not found at {model_path}")
    exit(1)

print(f"\n✓ Loading model from: {model_path}")

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ Using device: {device}")

# Load model
model = create_model('hybrid', pretrained=False)
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print("✓ Model loaded successfully")

# Get test dataloader
print("\n📁 Loading test dataset...")
_, _, test_loader = get_data_loaders(
    batch_size=config.BATCH_SIZE,
    num_workers=4
)

print(f"✓ Test samples: {len(test_loader.dataset)}")

# Evaluate
print("\n🔍 Running evaluation...\n")

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        
        all_probs.extend(probs.flatten().tolist())
        all_preds.extend(preds.flatten().tolist())
        all_labels.extend(labels.cpu().numpy().flatten().tolist())

# Calculate metrics
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_probs)

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp)
sensitivity = recall  # Same as recall

# Print results
print("\n" + "="*70)
print("📊 EVALUATION RESULTS")
print("="*70)
print(f"\n🎯 Overall Performance:")
print(f"   Accuracy:  {accuracy*100:.2f}%")
print(f"   Precision: {precision*100:.2f}%")
print(f"   Recall:    {recall*100:.2f}%")
print(f"   F1-Score:  {f1*100:.2f}%")
print(f"   AUC-ROC:   {auc*100:.2f}%")

print(f"\n🔬 Clinical Metrics:")
print(f"   Sensitivity (Recall):    {sensitivity*100:.2f}%")
print(f"   Specificity:             {specificity*100:.2f}%")

print(f"\n📋 Confusion Matrix:")
print(f"   True Negatives:  {tn}")
print(f"   False Positives: {fp}")
print(f"   False Negatives: {fn}")
print(f"   True Positives:  {tp}")

print(f"\n💡 Clinical Interpretation:")
print(f"   Of {tn + fp} NORMAL cases:")
print(f"     ✓ Correctly identified: {tn} ({specificity*100:.1f}%)")
print(f"     ✗ Misclassified:        {fp} ({fp/(tn+fp)*100:.1f}%)")
print(f"\n   Of {fn + tp} PNEUMONIA cases:")
print(f"     ✓ Correctly identified: {tp} ({sensitivity*100:.1f}%)")
print(f"     ✗ Missed (dangerous):   {fn} ({fn/(fn+tp)*100:.1f}%)")

print("\n" + "="*70)
print("💭 NOTES:")
print("="*70)
print("• This is the BASELINE model (before advanced techniques)")
print("• Current accuracy: 92.15% (from previous evaluation)")
print("• To improve further, retrain with:")
print("  - Focal Loss (handles class imbalance)")
print("  - Mixup/CutMix augmentation")
print("  - Label Smoothing")
print("  - Advanced LR scheduling")
print("  - Mixed Precision Training")
print("="*70 + "\n")
