# Robust and Explainable Pneumonia Detection from Chest X-rays

**Final Year B.Tech Project**

## Project Overview
Binary classification of chest X-rays into NORMAL vs PNEUMONIA using hybrid CNN–Vision Transformer models with lung-aware processing and explainability.

## Dataset
**Kaggle Chest X-ray Pneumonia Dataset**
- Download from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- Extract to: `data/chest_xray/`

### Expected Directory Structure
```
Pneumonia detection/
├── data/
│   └── chest_xray/
│       ├── train/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       ├── val/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       └── test/
│           ├── NORMAL/
│           └── PNEUMONIA/
├── config.py
├── dataset.py
├── requirements.txt
└── README.md
```

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Download dataset:**
   - Go to the Kaggle link above
   - Download and extract to `data/chest_xray/`

3. **Test dataset loading:**
```bash
python dataset.py
```

4. **Evaluate a trained model:**
```bash
python evaluate.py models/MODEL_DIR/best_model.pth MODEL_TYPE
```
Example:
```bash
python evaluate.py models/hybrid_20251219_120000/best_model.pth hybrid
```

## Project Pipeline

### ✓ Step 1: Dataset Loading and Preprocessing
- [x] Dataset loader with proper directory structure
- [x] Image preprocessing (resize, normalize)
- [x] Data augmentation with style randomization
- [x] Train/Val/Test data loaders

### ✓ Step 2: Lung Segmentation/Masking
- [x] Lung segmentation model (U-Net architecture)
- [x] CV-based fallback segmentation
- [x] Preprocessing with lung masks
- [x] Integration with dataset loader

### ✓ Step 3: Hybrid CNN–Transformer Model
- [x] CNN backbone (ResNet-18)
- [x] Vision Transformer integration (ViT-Base)
- [x] Hybrid architecture with feature fusion
- [x] Baseline models for ablation (CNN-only, ViT-only)

### ✓ Step 4: Training Pipeline
- [x] Training loop with BCEWithLogitsLoss
- [x] Validation loop
- [x] Early stopping
- [x] Model checkpointing
- [x] Learning rate scheduling
- [x] Metrics tracking (Loss, Acc, AUC, F1)
- [x] Training visualization

### ✓ Step 5: Evaluation
- [x] Comprehensive metrics (Acc, Prec, Recall, F1, AUC, Specificity)
- [x] Confusion matrix visualization
- [x] ROC curve plotting
- [x] Classification report
- [x] Test set evaluation
- [x] Model comparison utilities

### Step 6: Explainability (Grad-CAM)
- [ ] Grad-CAM implementation
- [ ] Heatmap generation
- [ ] Faithfulness evaluation
- [ ] Lung-overlap score

### Step 7: Ablation Study
- [ ] CNN only
- [ ] ViT only
- [ ] Hybrid CNN–ViT
- [ ] With vs without lung masking

## Current Status
✓ Step 1 completed: Dataset loading and preprocessing ready
✓ Step 2 completed: Lung segmentation implemented
✓ Step 3 completed: Hybrid CNN-ViT model architecture ready
✓ Step 4 completed: Training pipeline with early stopping and checkpointing
✓ Step 5 completed: Comprehensive evaluation with metrics and visualizations

## Configuration
All hyperparameters and paths are in `config.py`. Key parameters:
- Image size: 224×224
- Batch size: 32
- CNN backbone: ResNet-18
- Transformer: ViT-Base
- Learning rate: 1e-4

## Notes
- Single GPU or Google Colab compatible
- All code is modular and commented
- Designed for B.Tech project scope

## Authors
- **Aditya Chowdhary**
- **Ishan Dey**
- **Agnideep Ghorai**
