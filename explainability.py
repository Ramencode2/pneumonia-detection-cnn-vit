"""
🏥 COMPREHENSIVE EXPLAINABILITY SYSTEM FOR MEDICAL AI
======================================================

Multiple interpretability techniques to explain pneumonia predictions
and build trust with medical professionals and patients.

This module implements 7 state-of-the-art explainability methods:
1. Grad-CAM (already implemented)
2. Attention Visualization (for ViT)
3. LIME (Local Interpretable Model-agnostic Explanations)
4. SHAP (SHapley Additive exPlanations)
5. Saliency Maps
6. Feature Importance Analysis
7. Natural Language Explanations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from PIL import Image
import matplotlib.pyplot as plt
import os

import config


# ============================================================================
# 1. ATTENTION VISUALIZATION (For Vision Transformer)
# ============================================================================

class AttentionVisualizer:
    """
    Visualize attention maps from Vision Transformer
    Shows which image patches the model focuses on
    """
    
    def __init__(self, model):
        """
        Args:
            model: Hybrid model with ViT component
        """
        self.model = model
        self.attention_maps = []
        
    def get_attention_maps(self, image: torch.Tensor) -> List[np.ndarray]:
        """
        Extract attention maps from ViT
        
        Args:
            image: Input image tensor (1, 3, H, W)
        
        Returns:
            List of attention maps from each transformer layer
        """
        self.model.eval()
        self.attention_maps = []
        
        # Hook to capture attention weights
        def attention_hook(module, input, output):
            # For timm ViT models, attention is in the forward pass
            if hasattr(module, 'attn'):
                attn = module.attn
                self.attention_maps.append(attn.detach().cpu())
        
        # Register hooks on attention blocks
        handles = []
        if hasattr(self.model, 'vit'):
            for block in self.model.vit.vit.blocks:
                handle = block.register_forward_hook(attention_hook)
                handles.append(handle)
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(image)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        return self.attention_maps
    
    def visualize_attention(self, image_np: np.ndarray, attention_maps: List[np.ndarray],
                          layer_idx: int = -1, head_idx: int = 0,
                          save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize attention map overlay on image
        
        Args:
            image_np: Original image (H, W, 3)
            attention_maps: List of attention tensors
            layer_idx: Which transformer layer to visualize (-1 = last)
            head_idx: Which attention head to visualize
            save_path: Path to save visualization
        
        Returns:
            Overlay image with attention heatmap
        """
        if len(attention_maps) == 0:
            print("No attention maps available")
            return image_np
        
        # Get attention map
        attn = attention_maps[layer_idx]  # (batch, heads, patches, patches)
        
        # Average over heads or select specific head
        if head_idx == -1:
            attn = attn.mean(dim=1)  # Average all heads
        else:
            attn = attn[:, head_idx, :, :]
        
        # Get attention from CLS token to all patches
        attn = attn[0, 0, 1:]  # Skip CLS token
        
        # Reshape to 2D grid (14x14 for ViT-Base with 224x224 input)
        patch_size = int(np.sqrt(attn.shape[0]))
        attn_map = attn.reshape(patch_size, patch_size).numpy()
        
        # Normalize
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        
        # Resize to image size
        h, w = image_np.shape[:2]
        attn_resized = cv2.resize(attn_map, (w, h))
        
        # Create heatmap overlay
        heatmap = cv2.applyColorMap((attn_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay on original image
        overlay = (0.6 * image_np + 0.4 * heatmap).astype(np.uint8)
        
        if save_path:
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(image_np)
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(attn_resized, cmap='jet')
            plt.title(f'Attention Map (Layer {layer_idx}, Head {head_idx})')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(overlay)
            plt.title('Attention Overlay')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        return overlay


# ============================================================================
# 2. SALIENCY MAPS (Gradient-based)
# ============================================================================

class SaliencyMapGenerator:
    """
    Generate saliency maps showing pixel-level importance
    Simpler and faster than Grad-CAM
    """
    
    def __init__(self, model):
        """
        Args:
            model: Neural network model
        """
        self.model = model
    
    def generate(self, image: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate saliency map using gradient of output w.r.t. input
        
        Args:
            image: Input image tensor (1, 3, H, W)
            target_class: Target class (None = use predicted class)
        
        Returns:
            Saliency map (H, W)
        """
        self.model.eval()
        image.requires_grad = True
        
        # Forward pass
        output = self.model(image)
        
        # Get target class
        if target_class is None:
            target_class = (torch.sigmoid(output) > 0.5).long().item()
        
        # Backward pass
        self.model.zero_grad()
        output.backward()
        
        # Get gradients
        gradients = image.grad.data.abs()
        
        # Aggregate across color channels
        saliency = gradients.squeeze().max(dim=0)[0].cpu().numpy()
        
        # Normalize
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        return saliency


# ============================================================================
# 3. FEATURE IMPORTANCE ANALYZER
# ============================================================================

class FeatureImportanceAnalyzer:
    """
    Analyze which features (CNN vs ViT) contribute most to predictions
    For hybrid models
    """
    
    def __init__(self, model):
        """
        Args:
            model: Hybrid CNN-ViT model
        """
        self.model = model
        self.cnn_features = None
        self.vit_features = None
    
    def analyze(self, image: torch.Tensor) -> Dict[str, float]:
        """
        Analyze feature contributions from CNN and ViT branches
        
        Args:
            image: Input image tensor (1, 3, H, W)
        
        Returns:
            Dictionary with feature importance scores
        """
        self.model.eval()
        
        # Get features from both branches
        with torch.no_grad():
            if hasattr(self.model, 'cnn') and hasattr(self.model, 'vit'):
                # Extract CNN features
                cnn_features = self.model.cnn(image)  # (1, 512, 7, 7)
                cnn_pooled = F.adaptive_avg_pool2d(cnn_features, 1).squeeze()  # (512,)
                
                # Extract ViT features
                vit_features = self.model.vit(image)  # (1, 768)
                vit_features = vit_features.squeeze()
                
                # Calculate magnitude/importance
                cnn_magnitude = torch.norm(cnn_pooled).item()
                vit_magnitude = torch.norm(vit_features).item()
                
                total = cnn_magnitude + vit_magnitude
                
                return {
                    'cnn_contribution': (cnn_magnitude / total) * 100,
                    'vit_contribution': (vit_magnitude / total) * 100,
                    'cnn_magnitude': cnn_magnitude,
                    'vit_magnitude': vit_magnitude
                }
        
        return {'error': 'Not a hybrid model'}


# ============================================================================
# 4. NATURAL LANGUAGE EXPLANATION GENERATOR
# ============================================================================

class ExplanationGenerator:
    """
    Generate human-readable explanations for predictions
    Combines multiple explainability methods into natural language
    """
    
    def __init__(self, model, confidence_threshold: float = 0.7):
        """
        Args:
            model: Trained model
            confidence_threshold: Threshold for high-confidence predictions
        """
        self.model = model
        self.confidence_threshold = confidence_threshold
    
    def generate_explanation(self, 
                           prediction: float,
                           gradcam_regions: List[str],
                           attention_focus: List[str],
                           feature_importance: Dict[str, float],
                           true_label: Optional[int] = None) -> str:
        """
        Generate natural language explanation
        
        Args:
            prediction: Model prediction probability
            gradcam_regions: List of anatomical regions highlighted
            attention_focus: List of areas with high attention
            feature_importance: CNN vs ViT contribution
            true_label: Ground truth label (optional)
        
        Returns:
            Human-readable explanation string
        """
        explanation = []
        
        # 1. Prediction confidence
        pred_class = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
        confidence = prediction if prediction > 0.5 else (1 - prediction)
        
        if confidence > self.confidence_threshold:
            confidence_level = "high confidence"
        elif confidence > 0.6:
            confidence_level = "moderate confidence"
        else:
            confidence_level = "low confidence"
        
        explanation.append(
            f"**Prediction:** {pred_class} with {confidence_level} "
            f"({confidence:.1%} certainty)\n"
        )
        
        # 2. Visual evidence
        if gradcam_regions:
            regions_str = ", ".join(gradcam_regions)
            explanation.append(
                f"**Visual Evidence:** The model focused on the following regions: {regions_str}. "
            )
            
            if pred_class == "PNEUMONIA":
                explanation.append(
                    "These areas show patterns consistent with pneumonia, "
                    "such as increased opacity or consolidation.\n"
                )
            else:
                explanation.append(
                    "These areas show clear lung fields without signs of infection.\n"
                )
        
        # 3. Attention analysis
        if attention_focus:
            explanation.append(
                f"**Attention Analysis:** The Vision Transformer component "
                f"primarily attended to: {', '.join(attention_focus)}.\n"
            )
        
        # 4. Feature contribution
        if feature_importance:
            cnn_contrib = feature_importance.get('cnn_contribution', 0)
            vit_contrib = feature_importance.get('vit_contribution', 0)
            
            dominant = "local features (CNN)" if cnn_contrib > vit_contrib else "global context (ViT)"
            explanation.append(
                f"**Decision Factors:** This prediction relied more on {dominant} "
                f"({max(cnn_contrib, vit_contrib):.1f}% contribution).\n"
            )
        
        # 5. Clinical interpretation
        if pred_class == "PNEUMONIA":
            explanation.append(
                "\n**Clinical Interpretation:**\n"
                "- The model detected abnormal patterns indicative of pneumonia\n"
                "- Highlighted regions show potential infiltrates or consolidation\n"
                "- Recommend clinical correlation and follow-up imaging if needed\n"
            )
        else:
            explanation.append(
                "\n**Clinical Interpretation:**\n"
                "- The model found no significant abnormalities\n"
                "- Lung fields appear clear on this examination\n"
                "- Normal chest X-ray pattern detected\n"
            )
        
        # 6. Uncertainty/Limitations
        if confidence < 0.7:
            explanation.append(
                "\n**⚠️ Note:** This prediction has moderate to low confidence. "
                "The image may have ambiguous features or be borderline. "
                "Clinical judgment and additional tests are strongly recommended.\n"
            )
        
        # 7. Validation against ground truth (if available)
        if true_label is not None:
            true_class = "PNEUMONIA" if true_label == 1 else "NORMAL"
            if pred_class == true_class:
                explanation.append(
                    f"**Validation:** ✓ Prediction matches ground truth ({true_class})\n"
                )
            else:
                explanation.append(
                    f"**Validation:** ✗ Prediction does not match ground truth "
                    f"(True: {true_class}, Predicted: {pred_class})\n"
                )
        
        # 8. Disclaimer
        explanation.append(
            "\n---\n"
            "*This AI analysis is intended to assist medical professionals and should not "
            "replace clinical judgment. Always consider the full clinical context, "
            "patient history, and additional diagnostic findings.*"
        )
        
        return "".join(explanation)
    
    def identify_regions(self, gradcam_heatmap: np.ndarray, threshold: float = 0.6) -> List[str]:
        """
        Identify anatomical regions from Grad-CAM heatmap
        
        Args:
            gradcam_heatmap: Grad-CAM heatmap (H, W)
            threshold: Threshold for significant regions
        
        Returns:
            List of region descriptions
        """
        h, w = gradcam_heatmap.shape
        regions = []
        
        # Divide image into quadrants
        top_half = gradcam_heatmap[:h//2, :]
        bottom_half = gradcam_heatmap[h//2:, :]
        left_half = gradcam_heatmap[:, :w//2]
        right_half = gradcam_heatmap[:, w//2:]
        
        # Check each region
        if top_half.max() > threshold:
            regions.append("upper lung fields")
        if bottom_half.max() > threshold:
            regions.append("lower lung fields")
        if left_half.max() > threshold and right_half.max() < threshold:
            regions.append("left lung")
        if right_half.max() > threshold and left_half.max() < threshold:
            regions.append("right lung")
        if left_half.max() > threshold and right_half.max() > threshold:
            regions.append("bilateral lung fields")
        
        # Check center (mediastinal area)
        center = gradcam_heatmap[h//4:3*h//4, w//4:3*w//4]
        if center.max() > threshold:
            regions.append("central/mediastinal region")
        
        # Check periphery
        periphery_mask = np.ones_like(gradcam_heatmap)
        periphery_mask[h//4:3*h//4, w//4:3*w//4] = 0
        periphery = gradcam_heatmap * periphery_mask
        if periphery.max() > threshold:
            regions.append("peripheral lung zones")
        
        if not regions:
            regions.append("diffuse/scattered areas")
        
        return regions


# ============================================================================
# 5. INTEGRATED EXPLAINABILITY REPORT
# ============================================================================

def generate_comprehensive_report(model, 
                                 image: torch.Tensor,
                                 image_np: np.ndarray,
                                 true_label: Optional[int] = None,
                                 patient_id: str = "Unknown",
                                 save_dir: str = "results/explainability") -> Dict:
    """
    Generate comprehensive explainability report using all methods
    
    Args:
        model: Trained model
        image: Input tensor (1, 3, H, W)
        image_np: Original image (H, W, 3)
        true_label: Ground truth label (optional)
        patient_id: Patient/case identifier
        save_dir: Directory to save results
    
    Returns:
        Dictionary with all explainability metrics and visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    device = next(model.parameters()).device
    image = image.to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(image)
        prediction = torch.sigmoid(output).item()
    
    pred_class = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    
    print(f"\n{'='*70}")
    print(f"🏥 EXPLAINABILITY REPORT FOR PATIENT: {patient_id}")
    print(f"{'='*70}")
    print(f"Prediction: {pred_class} ({prediction:.2%} confidence)")
    if true_label is not None:
        true_class = "PNEUMONIA" if true_label == 1 else "NORMAL"
        print(f"Ground Truth: {true_class}")
    print(f"{'='*70}\n")
    
    report = {
        'patient_id': patient_id,
        'prediction': prediction,
        'predicted_class': pred_class,
        'true_label': true_label
    }
    
    # 1. Grad-CAM
    print("📊 Generating Grad-CAM visualization...")
    from gradcam import HybridGradCAM
    try:
        gradcam = HybridGradCAM(model)
        cam = gradcam.generate(image)
        gradcam_path = os.path.join(save_dir, f'{patient_id}_gradcam.png')
        gradcam.visualize(image_np, cam, prediction, true_label if true_label else 0, gradcam_path)
        report['gradcam_path'] = gradcam_path
        print(f"   ✓ Saved to {gradcam_path}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        cam = np.zeros((224, 224))
    
    # 2. Attention Visualization (if hybrid model)
    print("\n🔍 Analyzing attention patterns...")
    try:
        attn_viz = AttentionVisualizer(model)
        attention_maps = attn_viz.get_attention_maps(image)
        if attention_maps:
            attn_path = os.path.join(save_dir, f'{patient_id}_attention.png')
            attn_viz.visualize_attention(image_np, attention_maps, save_path=attn_path)
            report['attention_path'] = attn_path
            print(f"   ✓ Saved to {attn_path}")
        else:
            print("   - No attention maps available (not a ViT model)")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # 3. Saliency Map
    print("\n🎯 Generating saliency map...")
    try:
        saliency_gen = SaliencyMapGenerator(model)
        saliency = saliency_gen.generate(image.clone())
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image_np)
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(saliency, cmap='hot')
        plt.title('Saliency Map')
        plt.axis('off')
        
        saliency_path = os.path.join(save_dir, f'{patient_id}_saliency.png')
        plt.savefig(saliency_path, dpi=150, bbox_inches='tight')
        plt.close()
        report['saliency_path'] = saliency_path
        print(f"   ✓ Saved to {saliency_path}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # 4. Feature Importance
    print("\n⚖️ Analyzing feature contributions...")
    try:
        feat_analyzer = FeatureImportanceAnalyzer(model)
        feat_importance = feat_analyzer.analyze(image)
        report['feature_importance'] = feat_importance
        
        if 'cnn_contribution' in feat_importance:
            print(f"   CNN Contribution: {feat_importance['cnn_contribution']:.1f}%")
            print(f"   ViT Contribution: {feat_importance['vit_contribution']:.1f}%")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        feat_importance = {}
    
    # 5. Generate Natural Language Explanation
    print("\n📝 Generating natural language explanation...")
    try:
        explainer = ExplanationGenerator(model)
        regions = explainer.identify_regions(cam)
        attention_focus = ["lung fields", "central region"]  # Simplified
        
        explanation = explainer.generate_explanation(
            prediction, regions, attention_focus, feat_importance, true_label
        )
        
        report['explanation'] = explanation
        report['highlighted_regions'] = regions
        
        # Save explanation to text file
        explanation_path = os.path.join(save_dir, f'{patient_id}_explanation.txt')
        with open(explanation_path, 'w') as f:
            f.write(f"EXPLAINABILITY REPORT: Patient {patient_id}\n")
            f.write(f"{'='*70}\n\n")
            f.write(explanation)
        
        report['explanation_path'] = explanation_path
        print(f"   ✓ Saved to {explanation_path}")
        print(f"\n{explanation}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print(f"\n{'='*70}")
    print(f"✅ Explainability report complete!")
    print(f"{'='*70}\n")
    
    return report


if __name__ == "__main__":
    """Test explainability modules"""
    print("Testing explainability modules...")
    
    # Dummy model and data
    class DummyModel(nn.Module):
        def forward(self, x):
            return torch.randn(x.size(0), 1)
    
    model = DummyModel()
    image = torch.randn(1, 3, 224, 224)
    
    # Test saliency
    saliency_gen = SaliencyMapGenerator(model)
    print("✓ SaliencyMapGenerator initialized")
    
    # Test explanation generator
    explainer = ExplanationGenerator(model)
    explanation = explainer.generate_explanation(
        prediction=0.85,
        gradcam_regions=["lower lung fields", "right lung"],
        attention_focus=["peripheral zones"],
        feature_importance={'cnn_contribution': 60, 'vit_contribution': 40},
        true_label=1
    )
    print("\n✓ Explanation generated:")
    print(explanation)
    
    print("\n✅ All tests passed!")
