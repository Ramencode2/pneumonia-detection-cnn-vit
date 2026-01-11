"""
Advanced Data Augmentation Techniques for Pneumonia Detection
Implements state-of-the-art augmentation methods to boost accuracy
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import random
from typing import Tuple

import config


# ============================================================================
# Mixup and CutMix Augmentation
# ============================================================================

class MixupCutmix:
    """
    Implements Mixup and CutMix augmentation
    
    Mixup: Blend two images and their labels
    CutMix: Cut and paste patches between images
    
    Both proven to improve generalization in medical imaging
    """
    
    def __init__(self, 
                 mixup_alpha: float = 0.2,
                 cutmix_alpha: float = 1.0,
                 cutmix_prob: float = 0.5,
                 use_mixup: bool = True,
                 use_cutmix: bool = True):
        """
        Args:
            mixup_alpha: Mixup interpolation strength
            cutmix_alpha: CutMix interpolation strength
            cutmix_prob: Probability of applying CutMix vs Mixup
            use_mixup: Whether to use Mixup
            use_cutmix: Whether to use CutMix
        """
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_prob = cutmix_prob
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
    
    def mixup(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Apply Mixup augmentation
        
        Args:
            images: Batch of images (B, C, H, W)
            labels: Batch of labels (B,)
        
        Returns:
            mixed_images: Mixed images
            mixed_labels: Mixed labels  
            lam: Mixing coefficient
        """
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1.0
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        mixed_labels = lam * labels + (1 - lam) * labels[index]
        
        return mixed_images, mixed_labels, lam
    
    def cutmix(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Apply CutMix augmentation
        
        Args:
            images: Batch of images (B, C, H, W)
            labels: Batch of labels (B,)
        
        Returns:
            mixed_images: Mixed images
            mixed_labels: Mixed labels
            lam: Mixing coefficient
        """
        if self.cutmix_alpha > 0:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        else:
            lam = 1.0
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)
        
        _, _, H, W = images.size()
        
        # Generate random box
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)
        
        # Random center point
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        # Bounding box
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply CutMix
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual box size
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        mixed_labels = lam * labels + (1 - lam) * labels[index]
        
        return mixed_images, mixed_labels, lam
    
    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Mixup or CutMix randomly
        
        Args:
            images: Batch of images (B, C, H, W)
            labels: Batch of labels (B,)
        
        Returns:
            Augmented images and labels
        """
        if not self.use_mixup and not self.use_cutmix:
            return images, labels
        
        # Decide which augmentation to apply
        use_cutmix = self.use_cutmix and (random.random() < self.cutmix_prob)
        
        if use_cutmix:
            return self.cutmix(images, labels)[:2]
        elif self.use_mixup:
            return self.mixup(images, labels)[:2]
        else:
            return images, labels


# ============================================================================
# Advanced Image Transforms
# ============================================================================

class GaussianNoise(object):
    """Add Gaussian noise to images"""
    
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


class RandomErasing(object):
    """Randomly erase rectangular regions (similar to Cutout)"""
    
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
    
    def __call__(self, img):
        if random.random() > self.probability:
            return img
        
        for _ in range(100):
            area = img.size()[1] * img.size()[2]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
            
            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                img[:, x1:x1 + h, y1:y1 + w] = 0
                return img
        
        return img


def get_advanced_transforms(split: str = 'train') -> transforms.Compose:
    """
    Get advanced transformation pipeline
    
    Args:
        split: One of 'train', 'val', 'test'
    
    Returns:
        transforms.Compose object with advanced augmentations
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if split == 'train' and config.USE_ADVANCED_AUG:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomRotation(config.ROTATION_DEGREES),
            transforms.RandomHorizontalFlip(p=config.HORIZONTAL_FLIP_PROB),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # Random translation
                scale=(0.9, 1.1),  # Random scaling
                shear=10  # Random shear
            ),
            transforms.ColorJitter(
                brightness=config.BRIGHTNESS,
                contrast=config.CONTRAST,
                saturation=config.SATURATION
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.3),
            transforms.ToTensor(),
            RandomErasing(probability=0.25),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            normalize
        ])


# ============================================================================
# Test-Time Augmentation (TTA)
# ============================================================================

class TestTimeAugmentation:
    """
    Apply multiple augmentations during testing and average predictions
    Improves accuracy by 1-3% typically
    """
    
    def __init__(self, num_transforms: int = 5):
        """
        Args:
            num_transforms: Number of augmented versions to create
        """
        self.num_transforms = num_transforms
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def get_tta_transforms(self) -> list:
        """Get list of TTA transforms"""
        base_size = config.IMAGE_SIZE
        
        tta_transforms = [
            # Original
            transforms.Compose([
                transforms.Resize((base_size, base_size)),
                transforms.ToTensor(),
                self.normalize
            ]),
            # Horizontal flip
            transforms.Compose([
                transforms.Resize((base_size, base_size)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                self.normalize
            ]),
            # Slight rotation
            transforms.Compose([
                transforms.Resize((base_size, base_size)),
                transforms.RandomRotation(5),
                transforms.ToTensor(),
                self.normalize
            ]),
            # Brightness adjustment
            transforms.Compose([
                transforms.Resize((base_size, base_size)),
                transforms.ColorJitter(brightness=0.2),
                transforms.ToTensor(),
                self.normalize
            ]),
            # Slight scale
            transforms.Compose([
                transforms.Resize((int(base_size * 1.1), int(base_size * 1.1))),
                transforms.CenterCrop(base_size),
                transforms.ToTensor(),
                self.normalize
            ]),
        ]
        
        return tta_transforms[:self.num_transforms]
    
    def apply(self, model: nn.Module, image, device: torch.device) -> torch.Tensor:
        """
        Apply TTA and return averaged predictions
        
        Args:
            model: Trained model
            image: PIL Image
            device: Device to run on
        
        Returns:
            Average prediction across all augmentations
        """
        model.eval()
        predictions = []
        
        tta_transforms = self.get_tta_transforms()
        
        with torch.no_grad():
            for transform in tta_transforms:
                augmented = transform(image).unsqueeze(0).to(device)
                output = model(augmented)
                prob = torch.sigmoid(output)
                predictions.append(prob)
        
        # Average predictions
        avg_prediction = torch.stack(predictions).mean(dim=0)
        
        return avg_prediction


if __name__ == "__main__":
    """Test augmentation techniques"""
    print("Testing advanced augmentation...")
    
    # Test Mixup/CutMix
    mixup_cutmix = MixupCutmix(
        mixup_alpha=config.MIXUP_ALPHA,
        cutmix_alpha=config.CUTMIX_ALPHA,
        cutmix_prob=config.CUTMIX_PROB,
        use_mixup=config.USE_MIXUP,
        use_cutmix=config.USE_CUTMIX
    )
    
    # Dummy batch
    images = torch.randn(8, 3, 224, 224)
    labels = torch.randint(0, 2, (8, 1)).float()
    
    mixed_images, mixed_labels = mixup_cutmix(images, labels)
    print(f"✓ Mixup/CutMix: {images.shape} -> {mixed_images.shape}")
    
    # Test TTA
    tta = TestTimeAugmentation(num_transforms=config.TTA_TRANSFORMS)
    print(f"✓ TTA: {len(tta.get_tta_transforms())} transforms configured")
    
    print("\n✓ All augmentation tests passed!")
