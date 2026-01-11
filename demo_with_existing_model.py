"""
Evaluate existing model and generate comprehensive explanations
"""

import os
import torch
import numpy as np
from PIL import Image

print("\n" + "="*70)
print("📊 EVALUATING EXISTING MODEL WITH ADVANCED EXPLAINABILITY")
print("="*70)

# Use existing best model
model_path = "models/hybrid_20251219_144329/best_model.pth"

if not os.path.exists(model_path):
    print(f"\n❌ Model not found at {model_path}")
    print("\nAvailable models:")
    for root, dirs, files in os.walk("models"):
        for file in files:
            if file == "best_model.pth":
                print(f"  • {os.path.join(root, file)}")
else:
    print(f"\n✓ Found model: {model_path}")
    print("\n🔍 Generating explanations for test samples...")
    
    # Import after message
    from explainability import generate_comprehensive_report
    from model import create_model
    from dataset import get_transforms
    import config
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model('hybrid', pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded on {device}")
    
    # Find some test images
    test_normal_dir = os.path.join(config.DATA_DIR, 'test', 'NORMAL')
    test_pneumonia_dir = os.path.join(config.DATA_DIR, 'test', 'PNEUMONIA')
    
    # Get one of each
    normal_images = [f for f in os.listdir(test_normal_dir) if f.endswith(('.jpg', '.jpeg', '.png'))][:1]
    pneumonia_images = [f for f in os.listdir(test_pneumonia_dir) if f.endswith(('.jpg', '.jpeg', '.png'))][:1]
    
    transform = get_transforms('test', use_augmentation=False)
    
    output_dir = "results/explainability_demo"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📁 Saving results to: {output_dir}/\n")
    
    # Process normal case
    if normal_images:
        img_path = os.path.join(test_normal_dir, normal_images[0])
        print(f"[1/2] Processing NORMAL case: {normal_images[0]}")
        
        image_pil = Image.open(img_path).convert('RGB')
        image_np = np.array(image_pil.resize((224, 224)))
        image_tensor = transform(image_pil).unsqueeze(0)
        
        report = generate_comprehensive_report(
            model=model,
            image=image_tensor,
            image_np=image_np,
            true_label=0,
            patient_id="Normal_Case",
            save_dir=output_dir
        )
    
    # Process pneumonia case
    if pneumonia_images:
        img_path = os.path.join(test_pneumonia_dir, pneumonia_images[0])
        print(f"\n[2/2] Processing PNEUMONIA case: {pneumonia_images[0]}")
        
        image_pil = Image.open(img_path).convert('RGB')
        image_np = np.array(image_pil.resize((224, 224)))
        image_tensor = transform(image_pil).unsqueeze(0)
        
        report = generate_comprehensive_report(
            model=model,
            image=image_tensor,
            image_np=image_np,
            true_label=1,
            patient_id="Pneumonia_Case",
            save_dir=output_dir
        )
    
    print("\n" + "="*70)
    print("✅ EXPLAINABILITY DEMO COMPLETE!")
    print("="*70)
    print(f"\n📊 Generated files in {output_dir}/:")
    for file in sorted(os.listdir(output_dir)):
        print(f"  • {file}")
    
    print("\n💡 Next steps:")
    print("   1. View visualizations in results/explainability_demo/")
    print("   2. Read explanation reports (*.txt files)")
    print("   3. Use: python explain_prediction.py --image YOUR_IMAGE.jpg")
    print("="*70 + "\n")
