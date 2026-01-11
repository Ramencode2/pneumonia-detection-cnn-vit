"""
Hybrid CNN-Vision Transformer Model for Pneumonia Detection
Combines ResNet (CNN) with Vision Transformer for robust classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from typing import Optional

import config


# ============================================================================
# CNN Backbone (ResNet-18)
# ============================================================================

class CNNBackbone(nn.Module):
    """
    CNN backbone using pretrained ResNet-18
    Extracts local features and spatial patterns
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained ResNet-18
        resnet = models.resnet18(pretrained=pretrained)
        
        # Remove the final FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Output: (batch_size, 512, 7, 7) for 224x224 input
        self.out_channels = 512
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, 3, H, W)
        Returns:
            features: Feature maps (B, 512, 7, 7)
        """
        return self.features(x)


# ============================================================================
# Vision Transformer Backbone
# ============================================================================

class ViTBackbone(nn.Module):
    """
    Vision Transformer backbone using pretrained ViT
    Captures global context and long-range dependencies
    """
    
    def __init__(self, model_name: str = 'vit_base_patch16_224', pretrained: bool = True):
        super().__init__()
        
        # Load pretrained ViT from timm
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Get output dimension
        self.out_features = self.vit.num_features  # 768 for vit_base
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, 3, H, W)
        Returns:
            features: Global feature vector (B, 768)
        """
        return self.vit(x)


# ============================================================================
# Feature Fusion Module
# ============================================================================

class FeatureFusion(nn.Module):
    """
    Fuses CNN spatial features with ViT global features
    Uses adaptive pooling and concatenation
    """
    
    def __init__(self, cnn_channels: int = 512, vit_features: int = 768):
        super().__init__()
        
        # Global average pooling for CNN features
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Feature dimension after fusion
        self.fused_features = cnn_channels + vit_features
        
    def forward(self, cnn_features, vit_features):
        """
        Args:
            cnn_features: CNN output (B, 512, 7, 7)
            vit_features: ViT output (B, 768)
        Returns:
            fused: Concatenated features (B, 1280)
        """
        # Pool CNN features to global vector
        cnn_pooled = self.gap(cnn_features).squeeze(-1).squeeze(-1)  # (B, 512)
        
        # Concatenate CNN and ViT features
        fused = torch.cat([cnn_pooled, vit_features], dim=1)  # (B, 1280)
        
        return fused


# ============================================================================
# Classification Head
# ============================================================================

class ClassificationHead(nn.Module):
    """
    Binary classification head with dropout
    """
    
    def __init__(self, in_features: int, dropout: float = 0.3):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1)  # Binary classification
        )
    
    def forward(self, x):
        """
        Args:
            x: Fused features (B, in_features)
        Returns:
            logits: Raw scores (B, 1)
        """
        return self.classifier(x)


# ============================================================================
# Hybrid CNN-ViT Model
# ============================================================================

class HybridCNNViT(nn.Module):
    """
    Hybrid model combining CNN (ResNet-18) and Vision Transformer
    
    Architecture:
        Input (3, 224, 224)
            ├─> CNN Branch (ResNet-18) -> (512, 7, 7) -> GAP -> (512,)
            └─> ViT Branch (ViT-Base) -> (768,)
        Concatenate -> (1280,) -> Classification Head -> (1,)
    """
    
    def __init__(self, 
                 cnn_backbone: str = 'resnet18',
                 vit_backbone: str = 'vit_base_patch16_224',
                 pretrained: bool = True,
                 dropout: float = 0.3):
        """
        Args:
            cnn_backbone: CNN model name (currently only 'resnet18' supported)
            vit_backbone: ViT model name from timm
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate in classification head
        """
        super().__init__()
        
        # CNN branch
        self.cnn = CNNBackbone(pretrained=pretrained)
        
        # ViT branch
        self.vit = ViTBackbone(model_name=vit_backbone, pretrained=pretrained)
        
        # Fusion module
        self.fusion = FeatureFusion(
            cnn_channels=self.cnn.out_channels,
            vit_features=self.vit.out_features
        )
        
        # Classification head
        self.classifier = ClassificationHead(
            in_features=self.fusion.fused_features,
            dropout=dropout
        )
        
    def forward(self, x):
        """
        Forward pass through hybrid model
        
        Args:
            x: Input images (B, 3, 224, 224)
        Returns:
            logits: Raw scores (B, 1) - apply sigmoid for probabilities
        """
        # Extract features from both branches
        cnn_features = self.cnn(x)  # (B, 512, 7, 7)
        vit_features = self.vit(x)  # (B, 768)
        
        # Fuse features
        fused_features = self.fusion(cnn_features, vit_features)  # (B, 1280)
        
        # Classify
        logits = self.classifier(fused_features)  # (B, 1)
        
        return logits
    
    def get_gradcam_target_layer(self):
        """
        Returns the target layer for Grad-CAM visualization
        For ResNet-18, this is the last convolutional layer (layer4)
        """
        return self.cnn.features[-1]


# ============================================================================
# Baseline Models (for ablation study)
# ============================================================================

class CNNOnly(nn.Module):
    """
    CNN-only baseline (ResNet-18)
    For ablation study
    """
    
    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        
        self.cnn = CNNBackbone(pretrained=pretrained)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = ClassificationHead(
            in_features=self.cnn.out_channels,
            dropout=dropout
        )
    
    def forward(self, x):
        features = self.cnn(x)
        pooled = self.gap(features).squeeze(-1).squeeze(-1)
        logits = self.classifier(pooled)
        return logits
    
    def get_gradcam_target_layer(self):
        return self.cnn.features[-1]


class ViTOnly(nn.Module):
    """
    ViT-only baseline
    For ablation study
    """
    
    def __init__(self, 
                 model_name: str = 'vit_base_patch16_224',
                 pretrained: bool = True, 
                 dropout: float = 0.3):
        super().__init__()
        
        self.vit = ViTBackbone(model_name=model_name, pretrained=pretrained)
        self.classifier = ClassificationHead(
            in_features=self.vit.out_features,
            dropout=dropout
        )
    
    def forward(self, x):
        features = self.vit(x)
        logits = self.classifier(features)
        return logits


# ============================================================================
# Model Factory
# ============================================================================

def create_model(model_type: str = 'hybrid',
                 pretrained: bool = True,
                 dropout: float = config.DROPOUT) -> nn.Module:
    """
    Factory function to create different model variants
    
    Args:
        model_type: One of 'hybrid', 'cnn_only', 'vit_only'
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate
    
    Returns:
        model: PyTorch model
    """
    
    if model_type == 'hybrid':
        model = HybridCNNViT(
            cnn_backbone=config.CNN_BACKBONE,
            vit_backbone=config.TRANSFORMER_BACKBONE,
            pretrained=pretrained,
            dropout=dropout
        )
        print(f"Created Hybrid CNN-ViT model")
        print(f"  CNN: {config.CNN_BACKBONE}")
        print(f"  ViT: {config.TRANSFORMER_BACKBONE}")
        
    elif model_type == 'cnn_only':
        model = CNNOnly(pretrained=pretrained, dropout=dropout)
        print(f"Created CNN-only model ({config.CNN_BACKBONE})")
        
    elif model_type == 'vit_only':
        model = ViTOnly(
            model_name=config.TRANSFORMER_BACKBONE,
            pretrained=pretrained,
            dropout=dropout
        )
        print(f"Created ViT-only model ({config.TRANSFORMER_BACKBONE})")
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from 'hybrid', 'cnn_only', 'vit_only'")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model


def get_model_summary(model: nn.Module, input_size: tuple = (1, 3, 224, 224)):
    """
    Print model summary with layer details
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
    """
    try:
        from torchinfo import summary
        summary(model, input_size=input_size, device='cpu')
    except ImportError:
        print("Install torchinfo for detailed model summary: pip install torchinfo")
        print(f"\nModel: {model.__class__.__name__}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    """
    Test model creation and forward pass
    """
    print("="*70)
    print("TESTING MODEL CREATION")
    print("="*70)
    
    # Test Hybrid model
    print("\n1. Hybrid CNN-ViT Model:")
    print("-"*70)
    model_hybrid = create_model('hybrid', pretrained=False)  # pretrained=False for faster testing
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    print(f"\nInput shape: {x.shape}")
    
    output = model_hybrid(x)
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output.squeeze().detach().numpy()}")
    
    # Test CNN-only
    print("\n" + "="*70)
    print("2. CNN-Only Model:")
    print("-"*70)
    model_cnn = create_model('cnn_only', pretrained=False)
    output_cnn = model_cnn(x)
    print(f"Output shape: {output_cnn.shape}")
    
    # Test ViT-only
    print("\n" + "="*70)
    print("3. ViT-Only Model:")
    print("-"*70)
    model_vit = create_model('vit_only', pretrained=False)
    output_vit = model_vit(x)
    print(f"Output shape: {output_vit.shape}")
    
    print("\n" + "="*70)
    print("✓ All models created successfully!")
    print("="*70)
