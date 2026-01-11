"""
Ablation Study: Compare different model architectures and configurations
Tests the contribution of each component to overall performance
"""

import torch
import os
import json
from datetime import datetime
import numpy as np

from model import create_model
from dataset import get_data_loaders
from train import train_model, calculate_class_weights
from evaluate import evaluate_model
import config


def run_ablation_study(quick_test: bool = False):
    """
    Run comprehensive ablation study
    
    Tests:
    1. CNN-only vs ViT-only vs Hybrid
    2. With vs Without lung masking
    
    Args:
        quick_test: If True, run with reduced epochs for quick testing
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join('results', 'ablation', timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    # Calculate class weights
    pos_weight = calculate_class_weights()
    
    # Test configurations
    configurations = [
        # Architecture ablations (without lung masking)
        {'model_type': 'cnn_only', 'use_lung_mask': False, 'name': 'CNN-only'},
        {'model_type': 'vit_only', 'use_lung_mask': False, 'name': 'ViT-only'},
        {'model_type': 'hybrid', 'use_lung_mask': False, 'name': 'Hybrid'},
        
        # Lung masking ablation (with hybrid model)
        {'model_type': 'hybrid', 'use_lung_mask': True, 'name': 'Hybrid + Lung Mask'},
    ]
    
    # Adjust epochs for quick test
    num_epochs = 5 if quick_test else config.NUM_EPOCHS
    
    all_results = []
    
    print("="*80)
    print("ABLATION STUDY")
    print("="*80)
    print(f"Configurations to test: {len(configurations)}")
    print(f"Epochs per configuration: {num_epochs}")
    print(f"Results will be saved to: {results_dir}")
    print("="*80)
    
    for idx, config_dict in enumerate(configurations, 1):
        model_type = config_dict['model_type']
        use_lung_mask = config_dict['use_lung_mask']
        config_name = config_dict['name']
        
        print(f"\n{'='*80}")
        print(f"Configuration {idx}/{len(configurations)}: {config_name}")
        print(f"  Model Type: {model_type}")
        print(f"  Lung Masking: {use_lung_mask}")
        print(f"{'='*80}\n")
        
        try:
            # Train model using train_model function
            print(f"\nTraining {config_name}...")
            model_dir = os.path.join('models', f'ablation_{model_type}_{timestamp}')
            
            training_result = train_model(
                model_type=model_type,
                use_lung_mask=use_lung_mask,
                num_epochs=num_epochs,
                batch_size=config.BATCH_SIZE,
                device=device,
                save_dir=model_dir
            )
            
            # Get trained model and history
            trained_model = training_result['model']
            history = training_result
            
            # Load test data for evaluation
            lung_segmenter = None
            if use_lung_mask:
                from lung_segmentation import LungSegmenter
                lung_segmenter = LungSegmenter()
            
            _, _, test_loader = get_data_loaders(
                batch_size=config.BATCH_SIZE,
                num_workers=0,
                use_lung_mask=use_lung_mask,
                lung_segmenter=lung_segmenter
            )
            
            # Evaluate on test set
            print(f"\nEvaluating {config_name}...")
            metrics, predictions, targets = evaluate_model(
                model=trained_model,
                data_loader=test_loader,
                device=device
            )
            
            # Store results
            result = {
                'configuration': config_name,
                'model_type': model_type,
                'use_lung_mask': use_lung_mask,
                'num_epochs': num_epochs,
                'metrics': metrics,
                'model_dir': model_dir,
                'best_val_loss': history.get('best_val_loss'),
                'best_val_acc': history.get('best_val_acc')
            }
            all_results.append(result)
            
            # Print results
            print(f"\n{config_name} Results:")
            print(f"  Accuracy:    {metrics['accuracy']:.4f}")
            print(f"  Precision:   {metrics['precision']:.4f}")
            print(f"  Recall:      {metrics['recall']:.4f}")
            print(f"  F1 Score:    {metrics['f1']:.4f}")
            print(f"  AUC:         {metrics['auc']:.4f}")
            print(f"  Specificity: {metrics['specificity']:.4f}")
            
            # Save individual result
            result_path = os.path.join(results_dir, f'{model_type}_{"masked" if use_lung_mask else "unmasked"}_results.json')
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=4)
            
        except Exception as e:
            print(f"\n✗ Error in configuration {config_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save comprehensive results
    comprehensive_path = os.path.join(results_dir, 'ablation_study_results.json')
    with open(comprehensive_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    # Generate comparison report (only if we have results)
    if all_results:
        generate_comparison_report(all_results, results_dir)
    else:
        print("\n⚠ No successful configurations to compare.")
    
    print(f"\n{'='*80}")
    print("ABLATION STUDY COMPLETE")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*80}\n")
    
    return all_results


def generate_comparison_report(results, output_dir):
    """
    Generate a comparison report of all configurations
    
    Args:
        results: List of result dictionaries
        output_dir: Directory to save report
    """
    
    report_path = os.path.join(output_dir, 'comparison_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ABLATION STUDY - COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Summary table
        f.write("PERFORMANCE COMPARISON\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Configuration':<25} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'AUC':<8}\n")
        f.write("-"*80 + "\n")
        
        for result in results:
            config_name = result['configuration']
            metrics = result['metrics']
            f.write(f"{config_name:<25} "
                   f"{metrics['accuracy']:<8.4f} "
                   f"{metrics['precision']:<8.4f} "
                   f"{metrics['recall']:<8.4f} "
                   f"{metrics['f1']:<8.4f} "
                   f"{metrics['auc']:<8.4f}\n")
        
        f.write("\n\n")
        
        # Best performing configurations
        f.write("BEST PERFORMING CONFIGURATIONS\n")
        f.write("-"*80 + "\n")
        
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'specificity']
        
        for metric in metrics_to_compare:
            best_result = max(results, key=lambda x: x['metrics'][metric])
            f.write(f"\nBest {metric.capitalize()}: {best_result['configuration']}\n")
            f.write(f"  Value: {best_result['metrics'][metric]:.4f}\n")
        
        f.write("\n\n")
        
        # Architecture comparison (CNN vs ViT vs Hybrid)
        f.write("ARCHITECTURE COMPARISON (without lung masking)\n")
        f.write("-"*80 + "\n")
        
        cnn_result = next((r for r in results if r['model_type'] == 'cnn' and not r['use_lung_mask']), None)
        vit_result = next((r for r in results if r['model_type'] == 'vit' and not r['use_lung_mask']), None)
        hybrid_result = next((r for r in results if r['model_type'] == 'hybrid' and not r['use_lung_mask']), None)
        
        if cnn_result and vit_result and hybrid_result:
            f.write(f"\nCNN-only Accuracy:    {cnn_result['metrics']['accuracy']:.4f}\n")
            f.write(f"ViT-only Accuracy:    {vit_result['metrics']['accuracy']:.4f}\n")
            f.write(f"Hybrid Accuracy:      {hybrid_result['metrics']['accuracy']:.4f}\n")
            
            # Calculate improvements
            hybrid_acc = hybrid_result['metrics']['accuracy']
            cnn_acc = cnn_result['metrics']['accuracy']
            vit_acc = vit_result['metrics']['accuracy']
            
            if hybrid_acc > max(cnn_acc, vit_acc):
                improvement = hybrid_acc - max(cnn_acc, vit_acc)
                f.write(f"\nHybrid improves over best single architecture by: {improvement:.4f} ({improvement*100:.2f}%)\n")
        
        f.write("\n\n")
        
        # Lung masking impact
        f.write("LUNG MASKING IMPACT (hybrid model)\n")
        f.write("-"*80 + "\n")
        
        hybrid_no_mask = next((r for r in results if r['model_type'] == 'hybrid' and not r['use_lung_mask']), None)
        hybrid_with_mask = next((r for r in results if r['model_type'] == 'hybrid' and r['use_lung_mask']), None)
        
        if hybrid_no_mask and hybrid_with_mask:
            f.write(f"\nWithout Lung Masking: {hybrid_no_mask['metrics']['accuracy']:.4f}\n")
            f.write(f"With Lung Masking:    {hybrid_with_mask['metrics']['accuracy']:.4f}\n")
            
            diff = hybrid_with_mask['metrics']['accuracy'] - hybrid_no_mask['metrics']['accuracy']
            f.write(f"\nDifference: {diff:+.4f} ({diff*100:+.2f}%)\n")
            
            if diff > 0:
                f.write("→ Lung masking improves performance\n")
            else:
                f.write("→ Lung masking does not improve performance\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"\nComparison report saved to: {report_path}")
    
    # Also print to console
    with open(report_path, 'r') as f:
        print(f.read())


if __name__ == "__main__":
    """
    Run ablation study
    Set quick_test=True for faster testing with fewer epochs
    """
    import sys
    
    # Check command line arguments
    quick_test = '--quick' in sys.argv or '-q' in sys.argv
    
    if quick_test:
        print("Running quick ablation study (5 epochs per configuration)")
        print("For full study, run without --quick flag\n")
    else:
        print("Running full ablation study")
        print("This will take significant time. For quick test, use --quick flag\n")
    
    results = run_ablation_study(quick_test=quick_test)
