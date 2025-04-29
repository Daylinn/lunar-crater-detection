"""
Training script for lunar crater detection using YOLOv8.
This was developed as part of my research on automated crater detection.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CraterLoss(nn.Module):
    """Custom loss function for crater detection.
    
    I implemented a combination of focal loss and IoU loss to handle:
    1. Class imbalance (many non-crater regions)
    2. Precise localization of crater boundaries
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # Controls class imbalance
        self.gamma = gamma  # Focuses on hard examples
        
    def forward(self, pred, target):
        # Focal loss helps with class imbalance in crater detection
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        # Custom IoU loss for better crater boundary localization
        pred_boxes = pred[..., :4]
        target_boxes = target[..., :4]
        
        # Calculate intersection area
        x1 = torch.max(pred_boxes[..., 0], target_boxes[..., 0])
        y1 = torch.max(pred_boxes[..., 1], target_boxes[..., 1])
        x2 = torch.min(pred_boxes[..., 2], target_boxes[..., 2])
        y2 = torch.min(pred_boxes[..., 3], target_boxes[..., 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Calculate union area
        pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
        target_area = (target_boxes[..., 2] - target_boxes[..., 0]) * (target_boxes[..., 3] - target_boxes[..., 1])
        union = pred_area + target_area - intersection
        
        iou = intersection / (union + 1e-6)
        iou_loss = 1 - iou
        
        return focal_loss.mean() + iou_loss.mean()

def get_custom_augmentation():
    """Creates custom augmentation pipeline for lunar images.
    
    I designed this augmentation strategy specifically for lunar terrain:
    1. Brightness/contrast variations to simulate different lighting
    2. Noise to handle sensor artifacts
    3. Geometric transforms for viewpoint variations
    4. Elastic transforms to simulate terrain deformations
    """
    return A.Compose([
        # Simulate different lighting conditions
        A.RandomBrightnessContrast(p=0.5),
        # Add sensor noise
        A.GaussNoise(p=0.3),
        # Basic geometric transforms
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        # Advanced transforms for terrain variations
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
        ], p=0.3),
        # Standard normalization
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def main():
    """Main training function for the crater detection model.
    
    I chose YOLOv8m as it provided the best balance between performance and speed.
    The training configuration was optimized through multiple experiments.
    """
    # Initialize YOLOv8 medium model (best balance of performance/speed)
    model = YOLO('yolov8m.pt')
    
    # Training configuration based on experimental results
    custom_config = {
        'data': 'data.yaml',  # Dataset configuration
        'epochs': 150,        # Increased epochs for better convergence
        'batch': 16,          # Optimal batch size for my GPU
        'imgsz': 640,         # Standard YOLO input size
        'patience': 50,       # Early stopping to prevent overfitting
        'device': 'cpu',      # Training on CPU (GPU if available)
        'workers': 8,         # Parallel data loading
        'project': 'runs/detect',  # Save training artifacts
        'name': 'train',      # Experiment name
        'exist_ok': True,     # Allow overwriting
        'pretrained': True,   # Use ImageNet weights
        'optimizer': 'AdamW', # Better than SGD for this task
        'lr0': 0.001,        # Initial learning rate
        'lrf': 0.01,         # Final learning rate
        'momentum': 0.937,    # Optimizer momentum
        'weight_decay': 0.0005,  # L2 regularization
        'warmup_epochs': 5.0, # Learning rate warmup
        'warmup_momentum': 0.8,  # Warmup momentum
        'warmup_bias_lr': 0.1,   # Warmup bias learning rate
        'box': 7.5,          # Box loss weight
        'cls': 0.5,          # Classification loss weight
        'dfl': 1.5,          # Distribution focal loss weight
        'close_mosaic': 10,  # Disable mosaic augmentation
        'resume': False,     # Start fresh
        'amp': True,         # Mixed precision training
        'fraction': 1.0,     # Use full dataset
        'freeze': None,      # No frozen layers
        'multi_scale': False,# Fixed image size
        'seed': 0,           # Reproducibility
        'plots': True,       # Save training plots
        'verbose': True,     # Detailed output
        'save': True,        # Save best model
        'save_period': -1,   # Save all checkpoints
        'cache': False,      # No image caching
    }
    
    # Train the model
    results = model.train(**custom_config)
    
    # Print training results
    print("\nTraining Results:")
    print(f"Best mAP50: {results.results_dict['metrics/mAP50(B)']:.3f}")
    print(f"Best mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.3f}")
    print(f"Training time: {results.results_dict['train/epoch_time']:.2f} seconds per epoch")

if __name__ == "__main__":
    main() 