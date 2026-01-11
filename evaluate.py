"""
Model Evaluation Module
Comprehensive evaluation with metrics, confusion matrix, and ROC curves
"""

import os
import json
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from tqdm import tqdm

import config
from model import create_model
from dataset import get_data_loaders


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_model(model: nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  device: torch.device,
                  return_predictions: bool = False) -> Dict:
    """
    Evaluate model on a dataset
    
    Args:
        model: PyTorch model
        dataloader: Data loader
        device: Device to evaluate on
        return_predictions: Whether to return predictions
    
    Returns:
        Dictionary containing metrics and optionally predictions
    """
    model.eval()
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
            
            # Store results
            all_probs.extend(probs if isinstance(probs, np.ndarray) else [probs])
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to arrays
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs > 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
    }
    
    # Calculate specificity (True Negative Rate)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    if return_predictions:
        metrics['predictions'] = all_preds
        metrics['probabilities'] = all_probs
        metrics['labels'] = all_labels
    
    return metrics


def plot_confusion_matrix(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         class_names: list = None,
                         save_path: Optional[str] = None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save figure
    """
    if class_names is None:
        class_names = config.CLASS_NAMES
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    
    # Add count and percentage annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
            ax.text(j + 0.5, i + 0.5, text,
                   ha='center', va='center',
                   color='white' if cm[i, j] > cm.max() / 2 else 'black',
                   fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved confusion matrix to {save_path}")
    
    plt.show()


def plot_roc_curve(y_true: np.ndarray,
                  y_prob: np.ndarray,
                  save_path: Optional[str] = None):
    """
    Plot ROC curve
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        save_path: Path to save figure
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
            label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved ROC curve to {save_path}")
    
    plt.show()


def plot_metrics_comparison(metrics_dict: Dict[str, Dict],
                           save_path: Optional[str] = None):
    """
    Plot comparison of metrics across different models/settings
    
    Args:
        metrics_dict: Dictionary with model names as keys and metrics as values
        save_path: Path to save figure
    """
    # Extract metrics
    model_names = list(metrics_dict.keys())
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    # Prepare data
    data = {metric: [] for metric in metric_names}
    for model_name in model_names:
        for metric in metric_names:
            data[metric].append(metrics_dict[model_name][metric])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(model_names))
    width = 0.15
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (metric, color) in enumerate(zip(metric_names, colors)):
        offset = width * (i - 2)
        bars = ax.bar(x + offset, data[metric], width, label=metric.upper(), color=color)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved metrics comparison to {save_path}")
    
    plt.show()


def print_classification_report(y_true: np.ndarray,
                                y_pred: np.ndarray,
                                class_names: list = None):
    """
    Print detailed classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    if class_names is None:
        class_names = config.CLASS_NAMES
    
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    # Additional metrics
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("\nConfusion Matrix Breakdown:")
    print(f"  True Negatives (TN):  {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  True Positives (TP):  {tp}")
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    print(f"\nSensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity:          {specificity:.4f}")
    print("="*70)


# ============================================================================
# Full Evaluation Pipeline
# ============================================================================

def evaluate_trained_model(checkpoint_path: str,
                          model_type: str = 'hybrid',
                          use_test_set: bool = True,
                          device: str = 'cuda',
                          save_dir: Optional[str] = None):
    """
    Complete evaluation pipeline for a trained model
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_type: Type of model ('hybrid', 'cnn_only', 'vit_only')
        use_test_set: Whether to evaluate on test set (vs validation)
        device: Device to evaluate on
        save_dir: Directory to save results
    
    Returns:
        Dictionary containing all evaluation results
    """
    print("="*70)
    print(f"EVALUATING {model_type.upper()} MODEL")
    print("="*70)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create save directory
    if save_dir is None:
        save_dir = os.path.join(config.RESULTS_DIR, 'evaluation')
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model
    print("\nLoading model...")
    model = create_model(model_type, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Load data
    print("\nLoading dataset...")
    _, val_loader, test_loader = get_data_loaders(batch_size=config.BATCH_SIZE, num_workers=0)
    
    eval_loader = test_loader if use_test_set else val_loader
    split_name = 'Test' if use_test_set else 'Validation'
    
    # Evaluate
    print(f"\nEvaluating on {split_name} set...")
    results = evaluate_model(model, eval_loader, device, return_predictions=True)
    
    # Print metrics
    print("\n" + "-"*70)
    print("RESULTS")
    print("-"*70)
    print(f"Accuracy:   {results['accuracy']:.4f}")
    print(f"Precision:  {results['precision']:.4f}")
    print(f"Recall:     {results['recall']:.4f}")
    print(f"F1 Score:   {results['f1']:.4f}")
    print(f"AUC:        {results['auc']:.4f}")
    print(f"Specificity:{results['specificity']:.4f}")
    
    # Classification report
    print_classification_report(
        results['labels'],
        results['predictions'],
        config.CLASS_NAMES
    )
    
    # Plot confusion matrix
    print("\nGenerating visualizations...")
    plot_confusion_matrix(
        results['labels'],
        results['predictions'],
        config.CLASS_NAMES,
        save_path=os.path.join(save_dir, f'{model_type}_confusion_matrix.png')
    )
    
    # Plot ROC curve
    plot_roc_curve(
        results['labels'],
        results['probabilities'],
        save_path=os.path.join(save_dir, f'{model_type}_roc_curve.png')
    )
    
    # Save results
    results_to_save = {
        'model_type': model_type,
        'checkpoint_path': checkpoint_path,
        'split': split_name,
        'metrics': {
            'accuracy': float(results['accuracy']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1': float(results['f1']),
            'auc': float(results['auc']),
            'specificity': float(results['specificity'])
        }
    }
    
    results_path = os.path.join(save_dir, f'{model_type}_evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=4)
    print(f"\n✓ Saved evaluation results to {results_path}")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    
    return results_to_save


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    """
    Example: Evaluate a trained model
    """
    import sys
    
    # Check if checkpoint path is provided
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        model_type = sys.argv[2] if len(sys.argv) > 2 else 'hybrid'
    else:
        print("Usage: python evaluate.py <checkpoint_path> [model_type]")
        print("\nExample:")
        print("  python evaluate.py models/hybrid_20231219_120000/best_model.pth hybrid")
        print("\nSearching for latest checkpoint...")
        
        # Try to find latest checkpoint
        models_dir = config.MODELS_DIR
        if os.path.exists(models_dir):
            subdirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
            if subdirs:
                subdirs.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
                latest_dir = os.path.join(models_dir, subdirs[0])
                checkpoint_path = os.path.join(latest_dir, 'best_model.pth')
                
                if os.path.exists(checkpoint_path):
                    # Infer model type from directory name
                    model_type = subdirs[0].split('_')[0]
                    print(f"Found checkpoint: {checkpoint_path}")
                    print(f"Model type: {model_type}")
                else:
                    print("No checkpoint found.")
                    sys.exit(1)
            else:
                print("No trained models found.")
                sys.exit(1)
        else:
            print(f"Models directory not found: {models_dir}")
            sys.exit(1)
    
    # Evaluate model
    evaluate_trained_model(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        use_test_set=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
