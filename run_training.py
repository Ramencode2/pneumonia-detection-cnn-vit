"""
Main training script with proper error handling and progress monitoring
"""

import os
import sys
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from train import train_model


def main():
    """Main training function with error handling"""
    
    print("\n" + "="*70)
    print("🚀 PNEUMONIA DETECTION - ADVANCED TRAINING")
    print("="*70)
    
    # Check CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Device: {device.upper()}")
    
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("   ⚠️  Training on CPU (will be slower)")
    
    # Display enabled features
    print("\n✨ ADVANCED FEATURES ENABLED:")
    print("="*70)
    
    if config.USE_FOCAL_LOSS:
        print(f"  ✓ Focal Loss (α={config.FOCAL_ALPHA}, γ={config.FOCAL_GAMMA})")
    
    if config.USE_LABEL_SMOOTHING:
        print(f"  ✓ Label Smoothing (ε={config.LABEL_SMOOTHING})")
    
    if config.USE_MIXUP:
        print(f"  ✓ Mixup Augmentation (α={config.MIXUP_ALPHA})")
    
    if config.USE_CUTMIX:
        print(f"  ✓ CutMix Augmentation (α={config.CUTMIX_ALPHA})")
    
    if config.USE_ADVANCED_AUG:
        print(f"  ✓ Advanced Data Augmentation")
    
    if config.USE_COSINE_ANNEALING:
        print(f"  ✓ Cosine Annealing LR (T_max={config.COSINE_T_MAX})")
    
    if config.USE_WARMUP:
        print(f"  ✓ Learning Rate Warmup ({config.WARMUP_EPOCHS} epochs)")
    
    if config.USE_AMP:
        print(f"  ✓ Mixed Precision Training (2-3x speedup)")
    
    print("="*70)
    
    # Training configuration
    print("\n📊 TRAINING CONFIGURATION:")
    print("="*70)
    print(f"  Model Type: Hybrid CNN-ViT")
    print(f"  Epochs: {config.NUM_EPOCHS}")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  Learning Rate: {config.LEARNING_RATE}")
    print(f"  Image Size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
    print(f"  Early Stopping Patience: {config.EARLY_STOPPING_PATIENCE}")
    print("="*70)
    
    # Confirmation
    print("\n⏳ Starting training... (this may take a while)")
    print("   Press Ctrl+C to stop\n")
    
    try:
        # Train model
        results = train_model(
            model_type='hybrid',
            use_lung_mask=False,
            pretrained=False,
            num_epochs=config.NUM_EPOCHS,
            device=device
        )
        
        # Success!
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\n📈 BEST RESULTS:")
        print(f"   Validation Loss: {results['best_val_loss']:.4f}")
        print(f"   Validation AUC:  {results['best_val_auc']:.4f}")
        print(f"\n💾 Models saved to: {results['save_dir']}")
        print("\n📊 Next steps:")
        print("   1. Evaluate on test set: python evaluate.py")
        print("   2. Generate explanations: python explain_prediction.py --image test.jpg")
        print("   3. View results: python show_all_results.py")
        print("="*70 + "\n")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("⚠️  TRAINING INTERRUPTED BY USER")
        print("="*70)
        print("\nPartial results may have been saved.")
        print("You can resume training or start fresh.\n")
        return 1
        
    except Exception as e:
        print("\n\n" + "="*70)
        print("❌ ERROR DURING TRAINING")
        print("="*70)
        print(f"\nError: {str(e)}\n")
        
        # Show traceback for debugging
        import traceback
        print("Traceback:")
        traceback.print_exc()
        print("\n" + "="*70)
        
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
