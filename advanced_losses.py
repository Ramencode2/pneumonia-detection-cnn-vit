"""
Advanced Loss Functions for Medical Image Classification
Includes Focal Loss, Label Smoothing, and other techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Focal Loss
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard example mining
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Focuses training on hard, misclassified examples
    Proven effective in medical imaging with class imbalance
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, pos_weight: float = 1.0):
        """
        Args:
            alpha: Weighting factor for class balance (0-1)
            gamma: Focusing parameter (higher = more focus on hard examples)
            pos_weight: Weight for positive class
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Raw logits (B, 1)
            targets: Ground truth labels (B, 1)
        
        Returns:
            Focal loss value
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Binary cross-entropy
        bce = F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=torch.tensor([self.pos_weight]).to(inputs.device), reduction='none'
        )
        
        # Compute p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Compute alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal loss
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * bce
        
        return focal_loss.mean()


# ============================================================================
# Binary Cross-Entropy with Label Smoothing
# ============================================================================

class LabelSmoothingBCELoss(nn.Module):
    """
    Binary Cross-Entropy with Label Smoothing
    
    Prevents overconfidence and improves generalization
    Smooths labels: 0 -> epsilon, 1 -> 1-epsilon
    """
    
    def __init__(self, smoothing: float = 0.1, pos_weight: float = 1.0):
        """
        Args:
            smoothing: Label smoothing factor (0.0 to 0.5)
            pos_weight: Weight for positive class
        """
        super().__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Raw logits (B, 1)
            targets: Ground truth labels (B, 1)
        
        Returns:
            BCE loss with label smoothing
        """
        # Apply label smoothing
        # 0 -> smoothing, 1 -> 1 - smoothing
        smoothed_targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        
        # Compute BCE loss
        loss = F.binary_cross_entropy_with_logits(
            inputs, smoothed_targets,
            pos_weight=torch.tensor([self.pos_weight]).to(inputs.device)
        )
        
        return loss


# ============================================================================
# Combined Focal + Label Smoothing Loss
# ============================================================================

class FocalLabelSmoothingLoss(nn.Module):
    """
    Combines Focal Loss with Label Smoothing
    Best of both worlds: hard example mining + better generalization
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, 
                 smoothing: float = 0.1, pos_weight: float = 1.0):
        """
        Args:
            alpha: Focal loss alpha
            gamma: Focal loss gamma
            smoothing: Label smoothing factor
            pos_weight: Weight for positive class
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.pos_weight = pos_weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Raw logits (B, 1)
            targets: Ground truth labels (B, 1)
        
        Returns:
            Combined loss value
        """
        # Apply label smoothing
        smoothed_targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Binary cross-entropy with smoothed labels
        bce = F.binary_cross_entropy_with_logits(
            inputs, smoothed_targets,
            pos_weight=torch.tensor([self.pos_weight]).to(inputs.device),
            reduction='none'
        )
        
        # Compute p_t (using smoothed targets)
        p_t = probs * smoothed_targets + (1 - probs) * (1 - smoothed_targets)
        
        # Compute alpha_t
        alpha_t = self.alpha * smoothed_targets + (1 - self.alpha) * (1 - smoothed_targets)
        
        # Focal weighting
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * bce
        
        return focal_loss.mean()


def get_loss_function(pos_weight: float = 1.0) -> nn.Module:
    """
    Get the appropriate loss function based on config
    
    Args:
        pos_weight: Weight for positive class
    
    Returns:
        Loss function module
    """
    import config
    
    if config.USE_FOCAL_LOSS and config.USE_LABEL_SMOOTHING:
        print(f"Using Focal + Label Smoothing Loss (alpha={config.FOCAL_ALPHA}, "
              f"gamma={config.FOCAL_GAMMA}, smoothing={config.LABEL_SMOOTHING})")
        return FocalLabelSmoothingLoss(
            alpha=config.FOCAL_ALPHA,
            gamma=config.FOCAL_GAMMA,
            smoothing=config.LABEL_SMOOTHING,
            pos_weight=pos_weight
        )
    elif config.USE_FOCAL_LOSS:
        print(f"Using Focal Loss (alpha={config.FOCAL_ALPHA}, gamma={config.FOCAL_GAMMA})")
        return FocalLoss(
            alpha=config.FOCAL_ALPHA,
            gamma=config.FOCAL_GAMMA,
            pos_weight=pos_weight
        )
    elif config.USE_LABEL_SMOOTHING:
        print(f"Using Label Smoothing BCE Loss (smoothing={config.LABEL_SMOOTHING})")
        return LabelSmoothingBCELoss(
            smoothing=config.LABEL_SMOOTHING,
            pos_weight=pos_weight
        )
    else:
        print("Using standard BCE with Logits Loss")
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))


if __name__ == "__main__":
    """Test loss functions"""
    print("Testing loss functions...")
    
    # Dummy data
    inputs = torch.randn(8, 1, requires_grad=True)
    targets = torch.randint(0, 2, (8, 1)).float()
    
    # Test Focal Loss
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    loss = focal(inputs, targets)
    print(f"✓ Focal Loss: {loss.item():.4f}")
    
    # Test Label Smoothing BCE
    ls_bce = LabelSmoothingBCELoss(smoothing=0.1)
    loss = ls_bce(inputs, targets)
    print(f"✓ Label Smoothing BCE: {loss.item():.4f}")
    
    # Test Combined
    combined = FocalLabelSmoothingLoss(alpha=0.25, gamma=2.0, smoothing=0.1)
    loss = combined(inputs, targets)
    print(f"✓ Focal + Label Smoothing: {loss.item():.4f}")
    
    print("\n✓ All loss function tests passed!")
