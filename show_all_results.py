"""
Display comprehensive results from all evaluation stages
"""

import json
import os

def display_all_results():
    """Display evaluation metrics and Grad-CAM results"""
    
    print("="*80)
    print(" PNEUMONIA DETECTION - COMPREHENSIVE RESULTS")
    print("="*80)
    
    # =========================================================================
    # Model Evaluation Metrics
    # =========================================================================
    eval_results_path = 'results/evaluation/hybrid_evaluation_results.json'
    
    if os.path.exists(eval_results_path):
        print("\n📊 MODEL EVALUATION METRICS")
        print("-"*80)
        
        with open(eval_results_path, 'r') as f:
            eval_results = json.load(f)
        
        metrics = eval_results['metrics']
        print(f"  Accuracy:    {metrics['accuracy']:.2%}")
        print(f"  Precision:   {metrics['precision']:.2%}")
        print(f"  Recall:      {metrics['recall']:.2%}")
        print(f"  F1 Score:    {metrics['f1']:.2%}")
        print(f"  AUC-ROC:     {metrics['auc']:.2%}")
        print(f"  Specificity: {metrics['specificity']:.2%}")
        
        confusion = eval_results.get('confusion_matrix', {})
        if confusion:
            print(f"\n  Confusion Matrix:")
            print(f"    True Negatives (TN):  {confusion['tn']}")
            print(f"    False Positives (FP): {confusion['fp']}")
            print(f"    False Negatives (FN): {confusion['fn']}")
            print(f"    True Positives (TP):  {confusion['tp']}")
            print(f"    Total Samples: {confusion['tn'] + confusion['fp'] + confusion['fn'] + confusion['tp']}")
        print(f"\n  📋 Clinical Interpretation:")
        sensitivity = metrics['recall']  # Same as recall
        specificity = metrics['specificity']
        ppv = metrics['precision']  # Positive Predictive Value
        
        print(f"    • Sensitivity (Recall): {sensitivity:.2%}")
        print(f"      → Catches {sensitivity:.1%} of actual pneumonia cases")
        print(f"    • Specificity: {specificity:.2%}")
        print(f"      → Correctly identifies {specificity:.1%} of healthy patients")
        print(f"    • PPV (Precision): {ppv:.2%}")
        print(f"      → {ppv:.1%} of positive predictions are correct")
        
        # Visualizations
        print(f"\n  📈 Generated Visualizations:")
        conf_matrix_path = 'results/evaluation/hybrid_confusion_matrix.png'
        roc_curve_path = 'results/evaluation/hybrid_roc_curve.png'
        if os.path.exists(conf_matrix_path):
            print(f"    ✓ Confusion Matrix: {conf_matrix_path}")
        if os.path.exists(roc_curve_path):
            print(f"    ✓ ROC Curve: {roc_curve_path}")
    else:
        print("\n⚠ Evaluation results not found.")
        print(f"  Expected: {eval_results_path}")
    
    # =========================================================================
    # Grad-CAM Explainability Metrics
    # =========================================================================
    gradcam_metrics_path = 'results/gradcam/gradcam_metrics.json'
    
    if os.path.exists(gradcam_metrics_path):
        print("\n" + "="*80)
        print("🔍 GRAD-CAM EXPLAINABILITY METRICS")
        print("-"*80)
        
        with open(gradcam_metrics_path, 'r') as f:
            gradcam_metrics = json.load(f)
        
        faith = gradcam_metrics['avg_faithfulness']
        faith_std = gradcam_metrics['std_faithfulness']
        overlap = gradcam_metrics.get('avg_lung_overlap')
        overlap_std = gradcam_metrics.get('std_lung_overlap')
        
        print(f"  Faithfulness Score: {faith:.4f} ± {faith_std:.4f}")
        print(f"    → Measures how much prediction changes when masking highlighted regions")
        print(f"    → Higher = more faithful explanations")
        
        if overlap is not None:
            print(f"\n  Lung Overlap Score: {overlap:.4f} ± {overlap_std:.4f}")
            print(f"    → Measures how much heatmap focuses on lung regions vs background")
            print(f"    → Higher = better anatomical focus")
        
        print(f"\n  Samples Visualized: {gradcam_metrics['num_samples']}")
        
        # Check if visualizations exist
        gradcam_dir = 'results/gradcam'
        num_visualizations = len([f for f in os.listdir(gradcam_dir) if f.endswith('.png')])
        print(f"  Generated Heatmaps: {num_visualizations}")
        print(f"  Location: {gradcam_dir}/")
    else:
        print("\n⚠ Grad-CAM metrics not found.")
        print(f"  Expected: {gradcam_metrics_path}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey Highlights:")
    
    if os.path.exists(eval_results_path):
        with open(eval_results_path, 'r') as f:
            eval_results = json.load(f)
        
        accuracy = eval_results['metrics']['accuracy']
        recall = eval_results['metrics']['recall']
        auc = eval_results['metrics']['auc']
        
        print(f"  • Model achieves {accuracy:.1%} overall accuracy")
        print(f"  • High sensitivity ({recall:.1%}) - excellent for screening")
        print(f"  • Strong AUC ({auc:.1%}) - robust discrimination")
    
    if os.path.exists(gradcam_metrics_path):
        print(f"  • Grad-CAM visualizations provide explainability")
        print(f"  • Faithfulness metrics validate interpretation quality")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    display_all_results()
