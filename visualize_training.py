"""
Visualize training history and results
"""

import os
import json
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import config


def plot_training_history(history_path: str, save_path: Optional[str] = None):
    """
    Plot training and validation metrics
    
    Args:
        history_path: Path to training_history.json
        save_path: Optional path to save figure
    """
    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss over Epochs')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy over Epochs')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot AUC
    axes[1, 0].plot(epochs, history['train_auc'], 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(epochs, history['val_auc'], 'r-', label='Validation', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].set_title('AUC over Epochs')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot learning rate
    axes[1, 1].plot(epochs, history['learning_rates'], 'g-', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved training plot to {save_path}")
    
    plt.show()


def plot_metrics_summary(history_path: str, save_path: Optional[str] = None):
    """
    Plot summary of final metrics
    
    Args:
        history_path: Path to training_history.json
        save_path: Optional path to save figure
    """
    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Get final metrics
    final_train = {
        'Accuracy': history['train_acc'][-1],
        'AUC': history['train_auc'][-1]
    }
    
    final_val = {
        'Accuracy': history['val_acc'][-1],
        'AUC': history['val_auc'][-1]
    }
    
    # Create bar plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    metrics = list(final_train.keys())
    x = np.arange(len(metrics))
    width = 0.35
    
    train_values = list(final_train.values())
    val_values = list(final_val.values())
    
    bars1 = ax.bar(x - width/2, train_values, width, label='Train', color='skyblue')
    bars2 = ax.bar(x + width/2, val_values, width, label='Validation', color='salmon')
    
    ax.set_ylabel('Score')
    ax.set_title('Final Metrics Summary', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved metrics summary to {save_path}")
    
    plt.show()


def visualize_latest_training(models_dir: str = config.MODELS_DIR):
    """
    Visualize the most recent training run
    
    Args:
        models_dir: Directory containing model checkpoints
    """
    # Find latest training directory
    subdirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    
    if not subdirs:
        print("No training runs found in", models_dir)
        return
    
    # Sort by modification time
    subdirs.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
    latest_dir = os.path.join(models_dir, subdirs[0])
    
    print(f"Visualizing training from: {latest_dir}")
    
    history_path = os.path.join(latest_dir, 'training_history.json')
    
    if not os.path.exists(history_path):
        print(f"Training history not found at {history_path}")
        return
    
    # Create visualizations
    vis_dir = os.path.join(config.RESULTS_DIR, 'training_plots')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Plot training history
    plot_training_history(
        history_path,
        save_path=os.path.join(vis_dir, 'training_history.png')
    )
    
    # Plot metrics summary
    plot_metrics_summary(
        history_path,
        save_path=os.path.join(vis_dir, 'metrics_summary.png')
    )
    
    print(f"\n✓ Visualizations saved to {vis_dir}")


if __name__ == "__main__":
    """
    Visualize the latest training run
    """
    visualize_latest_training()
