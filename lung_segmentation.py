"""
Lung Segmentation for Chest X-rays
Uses a pretrained U-Net model to extract lung regions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
from typing import Tuple, Optional
import cv2

import config


# ============================================================================
# U-Net Architecture for Lung Segmentation
# ============================================================================

class DoubleConv(nn.Module):
    """Double Convolution block: (Conv -> BN -> ReLU) * 2"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """
    Lightweight U-Net for lung segmentation
    Architecture suitable for B.Tech project scope
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        
        # Encoder (downsampling)
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder (upsampling)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        # Output layer
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Output
        out = self.out(dec1)
        return torch.sigmoid(out)


# ============================================================================
# Lung Segmentation Pipeline
# ============================================================================

class LungSegmenter:
    """
    Lung segmentation pipeline for chest X-rays
    Can use either pretrained U-Net or traditional CV methods as fallback
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cuda'):
        """
        Args:
            model_path: Path to pretrained U-Net weights (optional)
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.use_deep_learning = False
        
        # Try to load pretrained model
        if model_path and os.path.exists(model_path):
            self.model = UNet(in_channels=1, out_channels=1).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self.model.eval()
            self.use_deep_learning = True
            print(f"✓ Loaded pretrained lung segmentation model from {model_path}")
        else:
            print("⚠ No pretrained model found. Using traditional CV-based segmentation.")
            print("  Note: For better results, train or download a pretrained U-Net model.")
    
    def segment_with_unet(self, image: np.ndarray) -> np.ndarray:
        """
        Segment lungs using pretrained U-Net
        
        Args:
            image: Grayscale image (H, W) or RGB (H, W, 3)
        
        Returns:
            Binary mask (H, W) with values in [0, 1]
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image
        
        # Normalize to [0, 1]
        image_gray = image_gray.astype(np.float32) / 255.0
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_gray).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            mask = self.model(image_tensor)
        
        # Convert back to numpy
        mask = mask.squeeze().cpu().numpy()
        
        return mask
    
    def segment_with_cv(self, image: np.ndarray) -> np.ndarray:
        """
        Segment lungs using traditional computer vision methods
        Simple but effective fallback method
        
        Args:
            image: Grayscale image (H, W) or RGB (H, W, 3)
        
        Returns:
            Binary mask (H, W) with values in [0, 1]
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image.copy()
        
        # Apply histogram equalization to improve contrast
        image_gray = cv2.equalizeHist(image_gray)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
        
        # Otsu's thresholding
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        
        # Close small holes
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Open to remove noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find largest connected components (lungs)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
        
        # Keep only the two largest components (left and right lung)
        # Component 0 is background
        if num_labels > 1:
            # Get areas of all components except background
            areas = stats[1:, cv2.CC_STAT_AREA]
            
            # Get indices of largest components
            if len(areas) >= 2:
                largest_indices = np.argsort(areas)[-2:] + 1  # +1 because we excluded background
            elif len(areas) == 1:
                largest_indices = [1]
            else:
                largest_indices = []
            
            # Create mask with only largest components
            mask = np.zeros_like(opened)
            for idx in largest_indices:
                mask[labels == idx] = 255
        else:
            mask = opened
        
        # Dilate slightly to ensure we capture lung boundaries
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        mask = cv2.dilate(mask, kernel_dilate, iterations=1)
        
        # Normalize to [0, 1]
        mask = mask.astype(np.float32) / 255.0
        
        return mask
    
    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Segment lungs from chest X-ray
        
        Args:
            image: Input image as numpy array (H, W) or (H, W, 3)
        
        Returns:
            Binary lung mask (H, W) with values in [0, 1]
        """
        if self.use_deep_learning:
            return self.segment_with_unet(image)
        else:
            return self.segment_with_cv(image)
    
    def apply_mask(self, image: np.ndarray, mask: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """
        Apply lung mask to image
        
        Args:
            image: Original image (H, W, 3)
            mask: Binary mask (H, W)
            alpha: Blending factor (1.0 = full masking, 0.0 = no masking)
        
        Returns:
            Masked image (H, W, 3)
        """
        # Ensure mask is the same size as image
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        
        # Ensure mask is in [0, 1]
        if mask.max() > 1.0:
            mask = mask / 255.0
        
        # Apply mask
        if len(image.shape) == 3:
            # RGB image
            mask_3ch = np.stack([mask] * 3, axis=-1)
        else:
            # Grayscale image
            mask_3ch = mask
        
        # Blend: masked_image = alpha * (image * mask) + (1 - alpha) * image
        masked_image = alpha * (image * mask_3ch) + (1 - alpha) * image
        
        return masked_image.astype(image.dtype)


# ============================================================================
# Utility Functions
# ============================================================================

def visualize_segmentation(image_path: str, segmenter: LungSegmenter, save_path: Optional[str] = None):
    """
    Visualize lung segmentation result
    
    Args:
        image_path: Path to chest X-ray image
        segmenter: LungSegmenter instance
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Segment lungs
    mask = segmenter.segment(image_rgb)
    
    # Apply mask
    masked_image = segmenter.apply_mask(image_rgb, mask, alpha=1.0)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Lung Mask')
    axes[1].axis('off')
    
    axes[2].imshow(masked_image.astype(np.uint8))
    axes[2].set_title('Masked Image')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {save_path}")
    
    plt.show()


def test_segmentation(data_dir: str = config.DATA_DIR, num_samples: int = 3):
    """
    Test lung segmentation on sample images
    
    Args:
        data_dir: Root directory of the dataset
        num_samples: Number of sample images to test
    """
    import matplotlib.pyplot as plt
    
    print("Testing lung segmentation...")
    
    # Initialize segmenter
    model_path = os.path.join(config.MODELS_DIR, 'lung_segmentation_unet.pth')
    segmenter = LungSegmenter(model_path=model_path)
    
    # Get sample images
    train_normal = os.path.join(data_dir, 'train', 'NORMAL')
    train_pneumonia = os.path.join(data_dir, 'train', 'PNEUMONIA')
    
    sample_images = []
    
    # Get NORMAL samples
    if os.path.exists(train_normal):
        normal_files = [f for f in os.listdir(train_normal) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for i in range(min(num_samples // 2, len(normal_files))):
            sample_images.append(('NORMAL', os.path.join(train_normal, normal_files[i])))
    
    # Get PNEUMONIA samples
    if os.path.exists(train_pneumonia):
        pneumonia_files = [f for f in os.listdir(train_pneumonia) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for i in range(min(num_samples // 2 + 1, len(pneumonia_files))):
            sample_images.append(('PNEUMONIA', os.path.join(train_pneumonia, pneumonia_files[i])))
    
    if not sample_images:
        print("⚠ No sample images found. Please download the dataset first.")
        return
    
    # Process and visualize
    results_dir = os.path.join(config.RESULTS_DIR, 'segmentation_samples')
    os.makedirs(results_dir, exist_ok=True)
    
    for i, (label, img_path) in enumerate(sample_images[:num_samples]):
        print(f"\nProcessing sample {i+1}/{num_samples} ({label})...")
        save_path = os.path.join(results_dir, f'segmentation_sample_{i+1}_{label}.png')
        visualize_segmentation(img_path, segmenter, save_path)
    
    print(f"\n✓ Segmentation test complete. Results saved to {results_dir}")


if __name__ == "__main__":
    """
    Test lung segmentation
    """
    print("="*70)
    print("Lung Segmentation for Pneumonia Detection")
    print("="*70)
    
    # Check if dataset exists
    if not os.path.exists(config.DATA_DIR):
        print(f"\n⚠ Dataset not found at {config.DATA_DIR}")
        print("Please download the dataset first and run dataset.py")
    else:
        # Test segmentation
        test_segmentation(num_samples=4)
