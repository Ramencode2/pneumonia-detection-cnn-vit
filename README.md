# Robust and Explainable Pneumonia Detection from Chest X-rays

**B.Tech Final Year Project**

## Project Overview

Binary classification of chest X-rays into NORMAL vs PNEUMONIA using a hybrid CNN-Vision Transformer architecture with lung-aware processing and multi-method explainability.

**Key Results (Test Set - 624 images):**

| Metric | Value |
|--------|-------|
| Accuracy | 88.14% |
| Precision | 93.89% |
| Recall | 86.67% |
| F1 Score | 90.13% |
| AUC-ROC | 95.87% |
| Specificity | 90.60% |

## Dataset

**Kaggle Chest X-ray Pneumonia Dataset**
- Download from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- Extract to: `data/chest_xray/`

### Expected Directory Structure
```
data/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
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

## Usage

### Train a model
```bash
python run_training.py
# or directly:
python train.py
```

### Evaluate a trained model
```bash
python evaluate.py models/MODEL_DIR/best_model.pth MODEL_TYPE
```
Example:
```bash
python evaluate.py models/hybrid_20260104_225847/best_model.pth hybrid
```

### Visualize training history
```bash
python visualize_training.py
```

### Explain predictions
```bash
# Single image
python explain_prediction.py --image path/to/xray.jpg

# Batch processing
python explain_prediction.py --batch --data-dir data/chest_xray/test/PNEUMONIA/ --max-samples 20

# Compare two cases
python explain_prediction.py --compare normal.jpg pneumonia.jpg
```

### View results
```bash
python show_all_results.py
```

## Architecture

### Hybrid CNN-ViT Model

The model combines two complementary approaches via feature fusion:

- **CNN Backbone (EfficientNet-B0):** Extracts local spatial features (edges, textures, consolidation patterns)
- **ViT Backbone (ViT-Base):** Captures global context and long-range dependencies across the image
- **Fusion Module:** Concatenates CNN and ViT feature vectors
- **Classification Head:** Binary classification with dropout regularization

Baseline models (CNN-only, ViT-only) are also available for ablation comparisons.

### Training Pipeline

- **Loss:** BCEWithLogitsLoss with class weighting (pos_weight=2.5)
- **Optimizer:** AdamW (lr=3e-4, weight_decay=1e-4)
- **LR Schedule:** Cosine annealing with warmup (3 epochs)
- **Regularization:** Early stopping (patience=10), gradient clipping, dropout
- **Optional:** Focal Loss, Label Smoothing, Mixup, CutMix, RandAugment (configurable in `config.py`)

### Explainability (7 Methods)

1. **Grad-CAM** - Visual heatmaps showing important regions
2. **Attention Visualization** - ViT attention patterns across layers/heads
3. **Saliency Maps** - Pixel-level importance via input gradients
4. **Feature Importance** - CNN vs ViT contribution analysis
5. **Natural Language Explanations** - Human-readable clinical reports
6. **Anatomical Region Identification** - Maps activations to lung regions
7. **Confidence Indicators** - Prediction probability with uncertainty flags

### Bias Mitigation

- **Lung Segmentation:** U-Net architecture + CV-based fallback (Otsu + morphology) to focus model attention on lung regions
- **Class Weighting:** Addresses training data imbalance (1,341 Normal vs 3,875 Pneumonia)

## Project Structure

```
pneumonia-detection-cnn-vit/
├── config.py                  # All hyperparameters and paths
├── dataset.py                 # Data loading, preprocessing, augmentation
├── model.py                   # Hybrid CNN-ViT, CNN-only, ViT-only models
├── train.py                   # Training loop with early stopping
├── evaluate.py                # Evaluation metrics and visualization
├── run_training.py            # Training entry point with error handling
├── explainability.py          # 7 explainability methods
├── gradcam.py                 # Grad-CAM heatmaps and faithfulness
├── explain_prediction.py      # CLI for single/batch image explanation
├── advanced_augmentation.py   # Mixup, CutMix, RandAugment, TTA
├── advanced_losses.py         # Focal Loss, Label Smoothing
├── lung_segmentation.py       # U-Net and CV-based lung segmentation
├── ablation_study.py          # CNN vs ViT vs Hybrid comparison
├── visualize_training.py      # Training history plots
├── show_all_results.py        # Display evaluation and Grad-CAM results
├── requirements.txt           # Python dependencies
├── FINAL_RESULTS.md           # Detailed performance report
├── models/                    # Saved model checkpoints
└── results/                   # Evaluation outputs and visualizations
    ├── evaluation/            # Metrics, confusion matrix, ROC curve
    ├── gradcam/               # Grad-CAM heatmap samples
    ├── training_plots/        # Training/validation curves
    ├── segmentation_samples/  # Lung segmentation examples
    └── explainability_demo/   # Example explanations
```

## Configuration

All hyperparameters are in `config.py`. Key parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Image size | 224x224 | Standard for pretrained models |
| Batch size | 32 | Adjust based on GPU memory |
| CNN backbone | EfficientNet-B0 | 1,280 output channels |
| Transformer | ViT-Base (patch16) | 768 features |
| Learning rate | 3e-4 | Optimized for EfficientNet |
| Epochs | 20 | With early stopping (patience=10) |
| Pos weight | 2.5 | Class imbalance correction |

## Requirements

- Python 3.8+
- PyTorch 2.0+
- timm 0.9+
- See `requirements.txt` for full list

**Hardware:**
- Minimum: 8GB RAM, CPU (slow but works)
- Recommended: GPU with 8GB+ VRAM
- Alternative: Google Colab / Kaggle Notebooks (free GPU)

## Authors

- **Aditya Chowdhary**
- **Ishan Dey**
- **Agnideep Ghorai**
