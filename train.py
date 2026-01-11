"""
Training pipeline for Pneumonia Detection
Includes training loop, validation, early stopping, and checkpointing
"""

import os
import time
import json
from datetime import datetime
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import config
from model import create_model
from dataset import get_data_loaders, calculate_class_weights

try:
    from advanced_losses import get_loss_function
    from advanced_augmentation import MixupCutmix
    ADVANCED_TRAINING = True
except ImportError:
    ADVANCED_TRAINING = False
    print("Warning: Advanced training modules not available")


# ============================================================================
# Training Utilities
# ============================================================================

class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve
    """
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for metrics like accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Args:
            score: Current metric value
        Returns:
            True if should stop training
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class MetricsTracker:
    """
    Track and compute metrics during training
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.targets = []
        self.losses = []
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor, loss: float):
        """
        Update metrics with batch results
        
        Args:
            preds: Predicted probabilities (after sigmoid)
            targets: Ground truth labels
            loss: Batch loss value
        """
        self.predictions.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        self.losses.append(loss)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics
        
        Returns:
            Dictionary of metric values
        """
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Binary predictions (threshold = 0.5)
        preds_binary = (preds > 0.5).astype(int)
        
        # For mixup/cutmix, targets may be continuous - convert to binary for classification metrics
        targets_binary = (targets > 0.5).astype(int)
        
        # Compute metrics
        metrics = {
            'loss': np.mean(self.losses),
            'accuracy': accuracy_score(targets_binary, preds_binary),
            'precision': precision_score(targets_binary, preds_binary, zero_division=0),
            'recall': recall_score(targets_binary, preds_binary, zero_division=0),
            'f1': f1_score(targets_binary, preds_binary, zero_division=0),
            'auc': roc_auc_score(targets_binary, preds) if len(np.unique(targets_binary)) > 1 else 0.0
        }
        
        return metrics


# ============================================================================
# Training Functions
# ============================================================================

def train_one_epoch(model: nn.Module,
                    dataloader: torch.utils.data.DataLoader,
                    criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    scaler=None,
                    mixup_cutmix=None) -> Dict[str, float]:
    """
    Train for one epoch
    
    Args:
        model: PyTorch model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        scaler: GradScaler for mixed precision (optional)
        mixup_cutmix: Mixup/CutMix augmentation (optional)
    
    Returns:
        Dictionary of training metrics
    """
    model.train()
    tracker = MetricsTracker()
    use_amp = scaler is not None
    use_mixup = mixup_cutmix is not None
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        
        # Apply Mixup/CutMix if enabled
        if use_mixup:
            images, labels = mixup_cutmix(images, labels)
        
        # Forward pass with mixed precision
        optimizer.zero_grad()
        
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Track metrics
        probs = torch.sigmoid(outputs)
        tracker.update(probs.detach(), labels.detach(), loss.item())
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Compute epoch metrics
    metrics = tracker.compute()
    
    return metrics


def validate(model: nn.Module,
            dataloader: torch.utils.data.DataLoader,
            criterion: nn.Module,
            device: torch.device,
            epoch: int,
            use_amp: bool = False) -> Dict[str, float]:
    """
    Validate the model
    
    Args:
        model: PyTorch model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        use_amp: Whether to use mixed precision
    
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    tracker = MetricsTracker()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Valid]')
    
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            # Forward pass with optional mixed precision
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Track metrics
            probs = torch.sigmoid(outputs)
            tracker.update(probs, labels, loss.item())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Compute epoch metrics
    metrics = tracker.compute()
    
    return metrics


# ============================================================================
# Main Training Loop
# ============================================================================

def train_model(model_type: str = 'hybrid',
                use_lung_mask: bool = False,
                pretrained: bool = True,
                num_epochs: int = config.NUM_EPOCHS,
                learning_rate: float = config.LEARNING_RATE,
                batch_size: int = config.BATCH_SIZE,
                device: str = 'cuda',
                save_dir: Optional[str] = None) -> Dict:
    """
    Complete training pipeline
    
    Args:
        model_type: 'hybrid', 'cnn_only', or 'vit_only'
        use_lung_mask: Whether to use lung segmentation
        pretrained: Whether to use pretrained weights
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        device: Device to train on
        save_dir: Directory to save checkpoints
    
    Returns:
        Dictionary containing training history and best metrics
    """
    
    # Setup
    print("="*70)
    print(f"TRAINING {model_type.upper()} MODEL")
    print("="*70)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create save directory
    if save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(config.MODELS_DIR, f'{model_type}_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save directory: {save_dir}")
    
    # Load data
    print("\n" + "-"*70)
    print("Loading dataset...")
    print("-"*70)
    
    lung_segmenter = None
    if use_lung_mask:
        from lung_segmentation import LungSegmenter
        lung_segmenter = LungSegmenter()
        print("✓ Lung segmentation enabled")
    
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=batch_size,
        use_lung_mask=use_lung_mask,
        lung_segmenter=lung_segmenter
    )
    
    # Calculate class weights for imbalanced dataset
    pos_weight = calculate_class_weights()
    
    # Create model
    print("\n" + "-"*70)
    print("Creating model...")
    print("-"*70)
    model = create_model(model_type, pretrained=pretrained)
    model = model.to(device)
    
    # Loss function with advanced techniques
    if ADVANCED_TRAINING:
        criterion = get_loss_function(pos_weight=pos_weight.item())
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        print(f"\nLoss: BCEWithLogitsLoss (pos_weight={pos_weight.item():.4f})")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.WEIGHT_DECAY
    )
    print(f"Optimizer: AdamW (lr={learning_rate}, weight_decay={config.WEIGHT_DECAY})")
    
    # Learning rate scheduler - use Cosine Annealing if enabled
    if config.USE_COSINE_ANNEALING:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.COSINE_T_MAX,
            eta_min=config.COSINE_ETA_MIN
        )
        print(f"Scheduler: CosineAnnealingLR (T_max={config.COSINE_T_MAX}, eta_min={config.COSINE_ETA_MIN})")
    else:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        print(f"Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)")
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        mode='min'
    )
    print(f"Early stopping: patience={config.EARLY_STOPPING_PATIENCE}")
    
    # Mixed precision training setup
    scaler = None
    use_amp = config.USE_AMP and device.type == 'cuda'
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
        print(f"Mixed Precision (AMP): Enabled (2-3x speedup)")
    else:
        print(f"Mixed Precision (AMP): Disabled")
    
    # Mixup/CutMix augmentation setup
    mixup_cutmix = None
    if ADVANCED_TRAINING and (config.USE_MIXUP or config.USE_CUTMIX):
        mixup_cutmix = MixupCutmix(
            mixup_alpha=config.MIXUP_ALPHA,
            cutmix_alpha=config.CUTMIX_ALPHA,
            cutmix_prob=config.CUTMIX_PROB,
            use_mixup=config.USE_MIXUP,
            use_cutmix=config.USE_CUTMIX
        )
        print(f"Mixup/CutMix: Enabled (mixup_alpha={config.MIXUP_ALPHA}, cutmix_alpha={config.CUTMIX_ALPHA})")
    else:
        print(f"Mixup/CutMix: Disabled")
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_auc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
        'learning_rates': []
    }
    
    best_val_loss = float('inf')
    best_val_auc = 0.0
    
    # Training loop
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-"*70)
        
        # Adjust learning rate for warmup
        if config.USE_WARMUP and epoch <= config.WARMUP_EPOCHS:
            warmup_lr = learning_rate * (epoch / config.WARMUP_EPOCHS)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            print(f"Warmup LR: {warmup_lr:.6f}")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scaler, mixup_cutmix
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, epoch, use_amp
        )
        
        # Update learning rate
        if config.USE_COSINE_ANNEALING:
            scheduler.step()
        else:
            scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_auc'].append(train_metrics['auc'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_auc'].append(val_metrics['auc'])
        history['learning_rates'].append(current_lr)
        
        # Print metrics
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
              f"AUC: {train_metrics['auc']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"AUC: {val_metrics['auc']:.4f}, F1: {val_metrics['f1']:.4f}")
        print(f"LR: {current_lr:.6f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_val_auc = val_metrics['auc']
            
            checkpoint = {
                'epoch': epoch,
                'model_type': model_type,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'history': history
            }
            
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
            print(f"✓ Saved best model (val_loss: {best_val_loss:.4f})")
        
        # Early stopping check
        if early_stopping(val_metrics['loss']):
            print(f"\n⚠ Early stopping triggered at epoch {epoch}")
            break
    
    # Training complete
    elapsed_time = time.time() - start_time
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Time elapsed: {elapsed_time/60:.2f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation AUC: {best_val_auc:.4f}")
    
    # Save training history
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"\n✓ Saved training history to {history_path}")
    
    # Save final model
    final_checkpoint = {
        'epoch': epoch,
        'model_type': model_type,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'history': history,
        'config': {
            'use_lung_mask': use_lung_mask,
            'pretrained': pretrained,
            'learning_rate': learning_rate,
            'batch_size': batch_size
        }
    }
    torch.save(final_checkpoint, os.path.join(save_dir, 'final_model.pth'))
    print(f"✓ Saved final model")
    
    return {
        'history': history,
        'best_val_loss': best_val_loss,
        'best_val_auc': best_val_auc,
        'save_dir': save_dir
    }


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    """
    Train the hybrid model with advanced techniques
    """
    import sys
    
    try:
        print("\n" + "="*70)
        print("🚀 TRAINING WITH ADVANCED TECHNIQUES")
        print("="*70)
        print("\nEnabled Features:")
        print("  ✓ Focal Loss + Label Smoothing")
        print("  ✓ Mixup/CutMix Augmentation")
        print("  ✓ Advanced Data Augmentation")
        print("  ✓ Cosine Annealing LR Schedule")
        print("  ✓ Learning Rate Warmup")
        print("  ✓ Mixed Precision (AMP)")
        print("="*70 + "\n")
        
        # Train hybrid model
        # Note: Using pretrained=False to avoid download interruptions
        # The model will learn from scratch with advanced techniques
        results = train_model(
            model_type='hybrid',
            use_lung_mask=False,
            pretrained=False,  # Avoid pretrained weight download issues
            num_epochs=config.NUM_EPOCHS,  # Use config value
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        print(f"Best Validation Loss: {results['best_val_loss']:.4f}")
        print(f"Best Validation AUC: {results['best_val_auc']:.4f}")
        print(f"Training results saved to: {results['save_dir']}")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
