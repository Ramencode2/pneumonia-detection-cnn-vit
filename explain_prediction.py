"""
🏥 INTERACTIVE EXPLAINABILITY INTERFACE
========================================

Easy-to-use script for generating comprehensive explanations
for any chest X-ray prediction.

Usage:
    python explain_prediction.py --image path/to/xray.jpg --model path/to/model.pth
    
    Or for batch processing:
    python explain_prediction.py --batch --data-dir path/to/images/
"""

import argparse
import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms

import config
from model import create_model
from dataset import get_transforms
from explainability import generate_comprehensive_report


def explain_single_image(image_path: str, 
                        model_path: str,
                        model_type: str = 'hybrid',
                        output_dir: str = 'results/explanations',
                        patient_id: str = None) -> None:
    """
    Generate comprehensive explanation for a single X-ray image
    
    Args:
        image_path: Path to chest X-ray image
        model_path: Path to trained model checkpoint
        model_type: Type of model ('hybrid', 'cnn_only', 'vit_only')
        output_dir: Directory to save explanations
        patient_id: Patient identifier (default: image filename)
    """
    
    print("\n" + "="*70)
    print("🏥 PNEUMONIA DETECTION - EXPLAINABILITY SYSTEM")
    print("="*70)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    model = create_model(model_type, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print("✓ Model loaded successfully")
    
    # Load and preprocess image
    print(f"\nLoading image: {image_path}")
    if not os.path.exists(image_path):
        print(f"✗ Error: Image not found at {image_path}")
        return
    
    # Original image for visualization
    image_pil = Image.open(image_path).convert('RGB')
    image_np = np.array(image_pil.resize((224, 224)))
    
    # Preprocessed image for model
    transform = get_transforms('test', use_augmentation=False)
    image_tensor = transform(image_pil).unsqueeze(0)
    
    # Extract patient ID
    if patient_id is None:
        patient_id = os.path.splitext(os.path.basename(image_path))[0]
    
    # Generate comprehensive report
    print(f"\nGenerating explainability report...")
    print("-"*70)
    
    report = generate_comprehensive_report(
        model=model,
        image=image_tensor,
        image_np=image_np,
        true_label=None,  # Unknown for new images
        patient_id=patient_id,
        save_dir=output_dir
    )
    
    # Summary
    print("\n" + "="*70)
    print("📊 EXPLANATION SUMMARY")
    print("="*70)
    print(f"Patient ID: {patient_id}")
    print(f"Prediction: {report['predicted_class']} ({report['prediction']:.2%})")
    print(f"\nFiles generated:")
    for key, value in report.items():
        if key.endswith('_path'):
            print(f"  • {key.replace('_path', '').title()}: {value}")
    
    if 'highlighted_regions' in report:
        print(f"\nHighlighted Regions: {', '.join(report['highlighted_regions'])}")
    
    print("\n" + "="*70)
    print(f"✅ Complete! All files saved to: {output_dir}/")
    print("="*70 + "\n")


def explain_batch(data_dir: str,
                 model_path: str,
                 model_type: str = 'hybrid',
                 output_dir: str = 'results/explanations',
                 max_samples: int = 10) -> None:
    """
    Generate explanations for multiple images
    
    Args:
        data_dir: Directory containing X-ray images
        model_path: Path to trained model checkpoint
        model_type: Type of model
        output_dir: Directory to save explanations
        max_samples: Maximum number of samples to process
    """
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [
        f for f in os.listdir(data_dir)
        if os.path.splitext(f.lower())[1] in image_extensions
    ][:max_samples]
    
    print(f"\nFound {len(image_files)} images to process")
    print("="*70)
    
    for idx, image_file in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] Processing: {image_file}")
        print("-"*70)
        
        image_path = os.path.join(data_dir, image_file)
        patient_id = f"batch_{idx:03d}_{os.path.splitext(image_file)[0]}"
        
        try:
            explain_single_image(
                image_path=image_path,
                model_path=model_path,
                model_type=model_type,
                output_dir=output_dir,
                patient_id=patient_id
            )
        except Exception as e:
            print(f"✗ Error processing {image_file}: {e}")
            continue
    
    print("\n" + "="*70)
    print(f"✅ Batch processing complete! Processed {len(image_files)} images")
    print(f"Results saved to: {output_dir}/")
    print("="*70 + "\n")


def compare_predictions(image_path1: str,
                       image_path2: str,
                       model_path: str,
                       model_type: str = 'hybrid',
                       output_dir: str = 'results/comparisons') -> None:
    """
    Compare explanations for two different images side-by-side
    
    Args:
        image_path1: Path to first X-ray
        image_path2: Path to second X-ray
        model_path: Path to trained model
        model_type: Type of model
        output_dir: Directory to save comparison
    """
    
    print("\n" + "="*70)
    print("🔍 COMPARATIVE ANALYSIS")
    print("="*70)
    
    # Process both images
    print("\n[1/2] Processing first image...")
    explain_single_image(
        image_path=image_path1,
        model_path=model_path,
        model_type=model_type,
        output_dir=output_dir,
        patient_id="compare_A"
    )
    
    print("\n[2/2] Processing second image...")
    explain_single_image(
        image_path=image_path2,
        model_path=model_path,
        model_type=model_type,
        output_dir=output_dir,
        patient_id="compare_B"
    )
    
    print("\n" + "="*70)
    print("✅ Comparison complete! Check the output directory for results")
    print("="*70 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Generate explainability reports for pneumonia detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python explain_prediction.py --image data/test.jpg --model models/best_model.pth
  
  # Batch processing
  python explain_prediction.py --batch --data-dir data/test_images/ --max-samples 20
  
  # Compare two images
  python explain_prediction.py --compare image1.jpg image2.jpg --model models/best_model.pth
        """
    )
    
    # Arguments
    parser.add_argument('--image', type=str, help='Path to single X-ray image')
    parser.add_argument('--model', type=str, 
                       default='models/hybrid_20251219_144329/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--model-type', type=str, default='hybrid',
                       choices=['hybrid', 'cnn_only', 'vit_only'],
                       help='Type of model')
    parser.add_argument('--output-dir', type=str, default='results/explanations',
                       help='Output directory for explanations')
    parser.add_argument('--patient-id', type=str, help='Patient identifier')
    
    # Batch processing
    parser.add_argument('--batch', action='store_true',
                       help='Process multiple images')
    parser.add_argument('--data-dir', type=str,
                       help='Directory containing images for batch processing')
    parser.add_argument('--max-samples', type=int, default=10,
                       help='Maximum number of samples to process in batch mode')
    
    # Comparison mode
    parser.add_argument('--compare', nargs=2, metavar=('IMAGE1', 'IMAGE2'),
                       help='Compare two images side-by-side')
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model):
        print(f"✗ Error: Model not found at {args.model}")
        print("\nAvailable models:")
        models_dir = 'models'
        if os.path.exists(models_dir):
            for model_dir in os.listdir(models_dir):
                model_path = os.path.join(models_dir, model_dir, 'best_model.pth')
                if os.path.exists(model_path):
                    print(f"  • {model_path}")
        return
    
    # Determine mode
    if args.compare:
        # Comparison mode
        compare_predictions(
            image_path1=args.compare[0],
            image_path2=args.compare[1],
            model_path=args.model,
            model_type=args.model_type,
            output_dir=args.output_dir
        )
    
    elif args.batch:
        # Batch mode
        if not args.data_dir:
            print("✗ Error: --data-dir required for batch processing")
            return
        
        explain_batch(
            data_dir=args.data_dir,
            model_path=args.model,
            model_type=args.model_type,
            output_dir=args.output_dir,
            max_samples=args.max_samples
        )
    
    elif args.image:
        # Single image mode
        explain_single_image(
            image_path=args.image,
            model_path=args.model,
            model_type=args.model_type,
            output_dir=args.output_dir,
            patient_id=args.patient_id
        )
    
    else:
        # No valid mode specified
        parser.print_help()
        print("\n✗ Error: Specify --image, --batch, or --compare")


if __name__ == "__main__":
    main()
