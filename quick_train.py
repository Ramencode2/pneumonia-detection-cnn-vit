"""
Quick training script to test improvements
Uses CNN-only model for faster initialization
"""

import torch
from train import train_model

if __name__ == "__main__":
    print("\n" + "="*70)
    print("QUICK TRAINING TEST - CNN ONLY MODEL")
    print("="*70)
    print("\nTraining with advanced features:")
    print("  ✓ Focal Loss")
    print("  ✓ Label Smoothing")
    print("  ✓ Mixup/CutMix Augmentation")
    print("  ✓ Advanced Data Augmentation")
    print("  ✓ Cosine Annealing LR")
    print("  ✓ Learning Rate Warmup")
    print("  ✓ Mixed Precision (AMP)")
    print("="*70 + "\n")
    
    # Train CNN-only model (faster than hybrid)
    results = train_model(
        model_type='cnn_only',  # Faster than hybrid
        use_lung_mask=False,
        pretrained=True,
        num_epochs=5,  # Quick test
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"Best Validation Loss: {results['best_val_loss']:.4f}")
    print(f"Best Validation AUC: {results['best_val_auc']:.4f}")
    print(f"Results saved to: {results['save_dir']}")
    print("="*70 + "\n")
