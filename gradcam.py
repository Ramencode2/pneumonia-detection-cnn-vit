"""
Grad-CAM Explainability for Pneumonia Detection
Provides visual explanations of model predictions
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional, List
from PIL import Image
import matplotlib.pyplot as plt
import os

import config


class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping)
    Generates visual explanations for CNN-based models
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: The neural network model
            target_layer: The layer to compute gradients from (should be last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Hook to save forward pass activations"""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for input image
        
        Args:
            input_image: Input tensor of shape (1, C, H, W)
            target_class: Target class for CAM (None = use predicted class)
        
        Returns:
            cam: Grad-CAM heatmap of shape (H, W), normalized to [0, 1]
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        # Get prediction if target_class not specified
        if target_class is None:
            target_class = (torch.sigmoid(output) > 0.5).long().item()
        
        # Backward pass
        self.model.zero_grad()
        output.backward()
        
        # Get gradients and activations
        gradients = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)
        
        # Global average pooling on gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)  # Only positive contributions
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def generate_cam_overlay(self, original_image: np.ndarray, cam: np.ndarray, 
                            alpha: float = 0.5) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on original image
        
        Args:
            original_image: Original image as numpy array (H, W, 3)
            cam: Grad-CAM heatmap (h, w)
            alpha: Transparency factor for overlay
        
        Returns:
            overlay: Image with heatmap overlay (H, W, 3)
        """
        # Resize CAM to match original image size
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Convert CAM to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Ensure original image is uint8
        if original_image.dtype != np.uint8:
            original_image = (original_image * 255).astype(np.uint8)
        
        # Overlay
        overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
        
        return overlay


class HybridGradCAM:
    """
    Grad-CAM for Hybrid CNN-ViT model
    Uses the CNN backbone's last convolutional layer
    """
    
    def __init__(self, model):
        """
        Args:
            model: HybridCNNViT model instance
        """
        self.model = model
        
        # Get the last conv layer from ResNet backbone
        # CNNBackbone.features is a Sequential of ResNet layers
        # For ResNet-18, the last conv layer is in layer4[-1].conv2
        # Access it through the Sequential: features[7] is layer4
        target_layer_idx = 7  # layer4 in ResNet
        self.target_layer = model.cnn.features[target_layer_idx][-1].conv2
        
        self.gradcam = GradCAM(model, self.target_layer)
    
    def generate(self, input_image: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Generate Grad-CAM heatmap"""
        return self.gradcam.generate_cam(input_image, target_class)
    
    def visualize(self, original_image: np.ndarray, cam: np.ndarray, 
                  prediction: float, label: int, save_path: Optional[str] = None) -> None:
        """
        Visualize Grad-CAM results
        
        Args:
            original_image: Original image (H, W, 3)
            cam: Grad-CAM heatmap
            prediction: Model prediction (probability)
            label: Ground truth label
            save_path: Path to save visualization
        """
        overlay = self.gradcam.generate_cam_overlay(original_image, cam, alpha=0.5)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(overlay)
        pred_class = 'PNEUMONIA' if prediction > 0.5 else 'NORMAL'
        true_class = 'PNEUMONIA' if label == 1 else 'NORMAL'
        axes[2].set_title(f'Pred: {pred_class} ({prediction:.2%})\nTrue: {true_class}')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved Grad-CAM visualization to {save_path}")
        
        plt.close()


def evaluate_faithfulness(model, image: torch.Tensor, cam: np.ndarray, 
                         original_pred: float, device: torch.device,
                         mask_ratio: float = 0.2) -> float:
    """
    Evaluate faithfulness of Grad-CAM by masking important regions
    
    Faithfulness measures how much the prediction changes when we mask
    the regions highlighted by Grad-CAM. Higher drop = more faithful.
    
    Args:
        model: Neural network model
        image: Input image tensor (1, C, H, W)
        cam: Grad-CAM heatmap (h, w)
        original_pred: Original prediction probability
        device: Device to run model on
        mask_ratio: Ratio of image to mask (top % of important pixels)
    
    Returns:
        confidence_drop: Drop in prediction confidence after masking
    """
    model.eval()
    
    # Resize CAM to match image size
    _, _, h, w = image.shape
    cam_resized = cv2.resize(cam, (w, h))
    
    # Get top mask_ratio% of pixels
    threshold = np.percentile(cam_resized, (1 - mask_ratio) * 100)
    mask = cam_resized > threshold
    
    # Create masked image (set important regions to zero/mean)
    masked_image = image.clone()
    for c in range(3):
        masked_image[0, c][mask] = 0  # or use mean value
    
    # Get prediction on masked image
    with torch.no_grad():
        masked_output = model(masked_image.to(device))
        masked_pred = torch.sigmoid(masked_output).item()
    
    # Calculate confidence drop
    confidence_drop = abs(original_pred - masked_pred)
    
    return confidence_drop


def calculate_lung_overlap(cam: np.ndarray, lung_mask: np.ndarray, 
                          threshold: float = 0.5) -> float:
    """
    Calculate overlap between Grad-CAM heatmap and lung mask
    
    Measures how much the model focuses on lung regions vs background.
    Higher overlap = better focus on relevant anatomy.
    
    Args:
        cam: Grad-CAM heatmap (h, w), values in [0, 1]
        lung_mask: Binary lung mask (H, W), values in {0, 1}
        threshold: Threshold for binarizing CAM
    
    Returns:
        overlap_score: Intersection over union between CAM and lung mask
    """
    # Resize CAM to match lung mask
    h, w = lung_mask.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    
    # Binarize CAM
    cam_binary = (cam_resized > threshold).astype(np.uint8)
    lung_binary = (lung_mask > 0).astype(np.uint8)
    
    # Calculate IoU
    intersection = np.logical_and(cam_binary, lung_binary).sum()
    union = np.logical_or(cam_binary, lung_binary).sum()
    
    if union == 0:
        return 0.0
    
    overlap_score = intersection / union
    
    return overlap_score


def generate_gradcam_for_samples(model, data_loader, device: torch.device,
                                 num_samples: int = 20, output_dir: str = 'results/gradcam',
                                 lung_segmenter=None) -> dict:
    """
    Generate Grad-CAM visualizations for sample images
    
    Args:
        model: Trained model
        data_loader: DataLoader for test set
        device: Device to run on
        num_samples: Number of samples to visualize
        output_dir: Directory to save visualizations
        lung_segmenter: Optional lung segmenter for overlap calculation
    
    Returns:
        metrics: Dictionary containing faithfulness and overlap metrics
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Grad-CAM
    gradcam = HybridGradCAM(model)
    
    # Metrics
    faithfulness_scores = []
    overlap_scores = []
    
    count = 0
    for batch_idx, (images, labels) in enumerate(data_loader):
        if count >= num_samples:
            break
        
        for i in range(images.size(0)):
            if count >= num_samples:
                break
            
            # Get single image
            image = images[i:i+1].to(device)
            label = labels[i].item()
            
            # Get prediction
            with torch.no_grad():
                output = model(image)
                pred = torch.sigmoid(output).item()
            
            # Generate Grad-CAM
            cam = gradcam.generate(image, target_class=None)
            
            # Evaluate faithfulness
            faith_score = evaluate_faithfulness(model, image, cam, pred, device)
            faithfulness_scores.append(faith_score)
            
            # Calculate lung overlap if segmenter provided
            if lung_segmenter is not None:
                # Get original image (denormalize)
                img_np = images[i].cpu().numpy().transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = (img_np * std + mean) * 255
                img_np = np.clip(img_np, 0, 255).astype(np.uint8)
                
                # Get lung mask
                lung_mask = lung_segmenter.segment(img_np)
                overlap = calculate_lung_overlap(cam, lung_mask)
                overlap_scores.append(overlap)
            else:
                img_np = np.zeros((224, 224, 3), dtype=np.uint8)
            
            # Visualize
            save_path = os.path.join(output_dir, f'gradcam_sample_{count:03d}.png')
            gradcam.visualize(img_np, cam, pred, label, save_path)
            
            count += 1
    
    # Calculate average metrics
    metrics = {
        'avg_faithfulness': np.mean(faithfulness_scores),
        'std_faithfulness': np.std(faithfulness_scores),
        'avg_lung_overlap': np.mean(overlap_scores) if overlap_scores else None,
        'std_lung_overlap': np.std(overlap_scores) if overlap_scores else None,
        'num_samples': len(faithfulness_scores)
    }
    
    print("\nGrad-CAM Metrics:")
    print(f"  Faithfulness: {metrics['avg_faithfulness']:.4f} ± {metrics['std_faithfulness']:.4f}")
    if metrics['avg_lung_overlap'] is not None:
        print(f"  Lung Overlap: {metrics['avg_lung_overlap']:.4f} ± {metrics['std_lung_overlap']:.4f}")
    print(f"  Samples processed: {metrics['num_samples']}")
    
    return metrics


if __name__ == "__main__":
    """
    Test Grad-CAM generation on a trained model
    """
    print("Testing Grad-CAM implementation...")
    
    from model import create_model
    from dataset import get_data_loaders
    from lung_segmentation import LungSegmenter
    import json
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    checkpoint_path = 'models/hybrid_20251219_144329/best_model.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please train a model first using train.py")
    else:
        print(f"\nLoading checkpoint: {checkpoint_path}")
        
        # Create model
        model = create_model('hybrid').to(device)
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Loaded model from epoch {checkpoint['epoch']}")
        
        # Load data
        print("\nLoading test data...")
        _, _, test_loader = get_data_loaders(batch_size=8, num_workers=0)
        
        # Initialize lung segmenter
        print("\nInitializing lung segmenter...")
        lung_segmenter = LungSegmenter()
        
        # Generate Grad-CAM visualizations
        print("\nGenerating Grad-CAM visualizations...")
        metrics = generate_gradcam_for_samples(
            model=model,
            data_loader=test_loader,
            device=device,
            num_samples=20,
            output_dir='results/gradcam',
            lung_segmenter=lung_segmenter
        )
        
        # Save metrics
        metrics_path = 'results/gradcam/gradcam_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"\nSaved metrics to {metrics_path}")
        print("\n✓ Grad-CAM generation complete!")
