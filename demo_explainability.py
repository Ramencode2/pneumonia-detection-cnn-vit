"""
Quick demo of explainability features
Run this to see how the system works
"""

import os
import torch
from explainability import ExplanationGenerator, SaliencyMapGenerator
import numpy as np

def demo_natural_language_explanation():
    """Demo the natural language explanation generator"""
    
    print("\n" + "="*70)
    print("🎯 DEMO: NATURAL LANGUAGE EXPLANATIONS")
    print("="*70)
    
    explainer = ExplanationGenerator(model=None, confidence_threshold=0.7)
    
    # Scenario 1: High confidence pneumonia
    print("\nScenario 1: High Confidence Pneumonia Detection")
    print("-"*70)
    
    explanation = explainer.generate_explanation(
        prediction=0.93,
        gradcam_regions=["lower right lung field", "peripheral lung zones"],
        attention_focus=["right lung", "bilateral lower fields"],
        feature_importance={'cnn_contribution': 68.5, 'vit_contribution': 31.5},
        true_label=1
    )
    print(explanation)
    
    # Scenario 2: Low confidence borderline case
    print("\n" + "="*70)
    print("\nScenario 2: Low Confidence (Borderline Case)")
    print("-"*70)
    
    explanation = explainer.generate_explanation(
        prediction=0.58,
        gradcam_regions=["scattered areas", "central/mediastinal region"],
        attention_focus=["diffuse pattern"],
        feature_importance={'cnn_contribution': 52.3, 'vit_contribution': 47.7},
        true_label=None
    )
    print(explanation)
    
    # Scenario 3: Clear normal case
    print("\n" + "="*70)
    print("\nScenario 3: Clear Normal Case")
    print("-"*70)
    
    explanation = explainer.generate_explanation(
        prediction=0.12,
        gradcam_regions=["bilateral lung fields"],
        attention_focus=["peripheral zones", "lung fields"],
        feature_importance={'cnn_contribution': 45.8, 'vit_contribution': 54.2},
        true_label=0
    )
    print(explanation)
    
    print("\n" + "="*70)
    print("✅ Natural language explanation demo complete!")
    print("="*70)


def demo_region_identification():
    """Demo anatomical region identification"""
    
    print("\n" + "="*70)
    print("🏥 DEMO: ANATOMICAL REGION IDENTIFICATION")
    print("="*70)
    
    explainer = ExplanationGenerator(model=None)
    
    # Create simulated heatmaps
    test_cases = [
        ("Upper right focus", create_heatmap_upper_right()),
        ("Lower bilateral", create_heatmap_lower_bilateral()),
        ("Central mediastinal", create_heatmap_central()),
        ("Diffuse pattern", create_heatmap_diffuse())
    ]
    
    for name, heatmap in test_cases:
        regions = explainer.identify_regions(heatmap, threshold=0.6)
        print(f"\n{name}:")
        print(f"  Identified regions: {', '.join(regions)}")
    
    print("\n" + "="*70)
    print("✅ Region identification demo complete!")
    print("="*70)


def create_heatmap_upper_right():
    """Create synthetic heatmap focused on upper right"""
    heatmap = np.zeros((224, 224))
    heatmap[30:80, 140:190] = 0.9  # Upper right
    return heatmap


def create_heatmap_lower_bilateral():
    """Create synthetic heatmap focused on lower bilateral"""
    heatmap = np.zeros((224, 224))
    heatmap[150:200, 30:80] = 0.85  # Lower left
    heatmap[150:200, 140:190] = 0.85  # Lower right
    return heatmap


def create_heatmap_central():
    """Create synthetic heatmap focused on center"""
    heatmap = np.zeros((224, 224))
    heatmap[80:144, 80:144] = 0.8  # Center
    return heatmap


def create_heatmap_diffuse():
    """Create synthetic diffuse heatmap"""
    heatmap = np.random.rand(224, 224) * 0.4
    heatmap[heatmap > 0.3] = 0.7
    return heatmap


def show_available_commands():
    """Show what users can do with the explainability system"""
    
    print("\n" + "="*70)
    print("📚 EXPLAINABILITY SYSTEM - QUICK REFERENCE")
    print("="*70)
    
    print("\n1️⃣  EXPLAIN SINGLE IMAGE:")
    print("   " + "-"*66)
    print("   python explain_prediction.py --image path/to/xray.jpg")
    print("   python explain_prediction.py --image data/chest_xray/test/NORMAL/sample.jpg")
    
    print("\n2️⃣  BATCH PROCESSING:")
    print("   " + "-"*66)
    print("   python explain_prediction.py --batch --data-dir data/chest_xray/test/PNEUMONIA/")
    
    print("\n3️⃣  COMPARE TWO CASES:")
    print("   " + "-"*66)
    print("   python explain_prediction.py --compare normal.jpg pneumonia.jpg")
    
    print("\n4️⃣  SPECIFY MODEL:")
    print("   " + "-"*66)
    print("   python explain_prediction.py --image test.jpg --model models/hybrid_*/best_model.pth")
    
    print("\n5️⃣  CUSTOM OUTPUT DIRECTORY:")
    print("   " + "-"*66)
    print("   python explain_prediction.py --image test.jpg --output-dir my_results/")
    
    print("\n📁 OUTPUT FILES GENERATED:")
    print("   " + "-"*66)
    print("   • [patient_id]_gradcam.png      - Grad-CAM visualization")
    print("   • [patient_id]_attention.png    - Attention map (if ViT)")
    print("   • [patient_id]_saliency.png     - Saliency map")
    print("   • [patient_id]_explanation.txt  - Full text report")
    
    print("\n" + "="*70)
    print("\n✨ TIP: Start with a single image to see all visualizations,")
    print("   then use batch mode for processing multiple cases!")
    print("\n" + "="*70)


def main():
    """Run all demos"""
    
    print("\n" + "="*70)
    print("🏥 EXPLAINABILITY SYSTEM - INTERACTIVE DEMO")
    print("="*70)
    print("\nThis demo shows how the explainability system works")
    print("without needing actual model weights or images.")
    print("="*70)
    
    # Run demos
    demo_natural_language_explanation()
    demo_region_identification()
    show_available_commands()
    
    print("\n" + "="*70)
    print("🎓 WHAT THIS SOLVES:")
    print("="*70)
    print("""
The Black Box Problem in Medical AI:
    
❌ BEFORE (Black Box):
   Doctor: "Why did the AI say pneumonia?"
   AI: "..." (no answer, just a number)
   Result: No trust, no adoption

✅ AFTER (Explainable):
   Doctor: "Why did the AI say pneumonia?"
   AI: "I detected abnormal patterns in the lower right lung field
        showing increased opacity consistent with consolidation.
        My confidence is 93% based on local CNN features (68%) and
        global ViT context (32%). Here's a heatmap showing exactly
        where I'm looking..."
   Result: Trust, understanding, clinical validation

KEY BENEFITS:
    1. Shows WHERE the model is looking (Grad-CAM, Attention)
    2. Explains WHY in human language
    3. Identifies WHAT anatomical regions
    4. Provides CONFIDENCE levels
    5. Flags UNCERTAINTY for review
    6. Generates DOCUMENTATION for records
    
REGULATORY COMPLIANCE:
    ✓ FDA requirements for medical AI
    ✓ Audit trails
    ✓ Transparent decision-making
    ✓ Patient right to explanation
    """)
    
    print("="*70)
    print("\n💡 NEXT STEPS:")
    print("="*70)
    print("""
1. Read EXPLAINABILITY_GUIDE.py for complete documentation
2. Try: python explain_prediction.py --image [your_xray.jpg]
3. Review generated visualizations in results/explanations/
4. Customize explanations for your use case
5. Validate with medical professionals
    """)
    
    print("="*70)
    print("✅ Demo complete! Your AI is no longer a black box!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
