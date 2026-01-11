"""
Display evaluation results
"""

import json
import os
import config

results_dir = os.path.join(config.RESULTS_DIR, 'evaluation')

if os.path.exists(results_dir):
    json_files = [f for f in os.listdir(results_dir) if f.endswith('_evaluation_results.json')]
    
    if json_files:
        print("="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        
        for json_file in json_files:
            file_path = os.path.join(results_dir, json_file)
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            print(f"\nModel: {data['model_type'].upper()}")
            print(f"Split: {data['split']}")
            print("-"*70)
            print("Metrics:")
            for metric, value in data['metrics'].items():
                print(f"  {metric.capitalize():<15}: {value:.4f}")
            print()
    else:
        print("No evaluation results found.")
        print("Run: python evaluate.py")
else:
    print(f"Evaluation directory not found: {results_dir}")
    print("Run: python evaluate.py")

print("="*70)
