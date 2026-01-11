"""
Configuration file for Pneumonia Detection Project
Contains all hyperparameters and paths
"""

import os

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'chest_xray')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')

# Create directories if they don't exist
for dir_path in [RESULTS_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ============================================================================
# DATA PARAMETERS
# ============================================================================
IMAGE_SIZE = 224  # Standard size for pretrained models
BATCH_SIZE = 32
NUM_WORKERS = 8  # For DataLoader (increased for faster loading)
NUM_CLASSES = 1  # Binary classification (sigmoid output)

# DataLoader optimizations
PIN_MEMORY = True  # Faster data transfer to GPU
PERSISTENT_WORKERS = True  # Keep workers alive between epochs

# Class names
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

# ============================================================================
# MODEL PARAMETERS
# ============================================================================
# CNN Backbone options: 'resnet18', 'efficientnet_b0'
CNN_BACKBONE = 'resnet18'

# Transformer options: 'vit_base_patch16_224', 'swin_tiny_patch4_window7_224'
TRANSFORMER_BACKBONE = 'vit_base_patch16_224'

# Dropout
DROPOUT = 0.3

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 20  # Good balance between speed and convergence
EARLY_STOPPING_PATIENCE = 7  # Reasonable patience

# Mixed precision training (2-3x speedup, no accuracy loss)
USE_AMP = True  # Automatic Mixed Precision

# Advanced training techniques
USE_FOCAL_LOSS = True  # Focal loss for hard example mining
FOCAL_ALPHA = 0.25  # Focal loss alpha parameter
FOCAL_GAMMA = 2.0  # Focal loss gamma (focus on hard examples)

USE_LABEL_SMOOTHING = True  # Label smoothing for better generalization
LABEL_SMOOTHING = 0.1  # Smoothing factor

USE_COSINE_ANNEALING = True  # Cosine annealing LR schedule
COSINE_T_MAX = 20  # Number of epochs for cosine schedule (match NUM_EPOCHS)
COSINE_ETA_MIN = 1e-6  # Minimum learning rate

USE_WARMUP = True  # Learning rate warmup
WARMUP_EPOCHS = 3  # Number of warmup epochs

# Loss function
POS_WEIGHT = 1.0  # Adjust for class imbalance if needed

# Test-time augmentation (TTA) for inference
USE_TTA = True  # Apply multiple augmentations during testing
TTA_TRANSFORMS = 5  # Number of TTA transforms to average

# ============================================================================
# AUGMENTATION PARAMETERS
# ============================================================================
# Basic augmentation
ROTATION_DEGREES = 15
HORIZONTAL_FLIP_PROB = 0.5

# Style randomization (for bias mitigation)
BRIGHTNESS = 0.3
CONTRAST = 0.3
SATURATION = 0.15

# Advanced augmentation techniques
USE_ADVANCED_AUG = True  # RandAugment, AutoAugment-style transforms
RAND_AUG_N = 2  # Number of augmentation transformations
RAND_AUG_M = 9  # Magnitude of augmentations (0-10)
USE_MIXUP = True  # Mixup augmentation (proven for medical imaging)
MIXUP_ALPHA = 0.2  # Mixup interpolation strength
USE_CUTMIX = True  # CutMix augmentation
CUTMIX_ALPHA = 1.0  # CutMix interpolation strength
CUTMIX_PROB = 0.5  # Probability of applying CutMix

# ============================================================================
# LUNG SEGMENTATION PARAMETERS
# ============================================================================
USE_LUNG_SEGMENTATION = False  # Set to True to enable lung masking
LUNG_MASK_ALPHA = 1.0  # Blending factor for lung mask (1.0 = full masking)

# ============================================================================
# EXPLAINABILITY PARAMETERS
# ============================================================================
GRADCAM_LAYER = 'layer4'  # For ResNet-18
NUM_GRADCAM_SAMPLES = 50  # Number of samples for explainability evaluation

# Faithfulness evaluation
MASK_RATIOS = [0.1, 0.3, 0.5, 0.7]  # Percentage of image to mask

# ============================================================================
# DEVICE
# ============================================================================
DEVICE = 'cuda'  # Will be set to 'cpu' if CUDA not available

# ============================================================================
# RANDOM SEED
# ============================================================================
RANDOM_SEED = 42
