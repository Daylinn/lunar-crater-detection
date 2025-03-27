import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class CraterDataset(Dataset):
    """Dataset class for crater detection using YOLO format with crater-specific augmentations."""
    
    def __init__(self, images_dir, transform=None):
        """
        Initialize the dataset.
        
        Args:
            images_dir (str or Path): Directory containing crater images
            transform (callable, optional): Transform to be applied on images
        """
        self.images_dir = Path(images_dir)
        self.image_files = sorted([f for f in self.images_dir.glob("*.jpg")])
        self.transform = transform or self.get_default_transform()
        
        # Generate YOLO format annotations
        self.annotations = self._generate_annotations()
    
    def _generate_annotations(self):
        """Generate YOLO format annotations for each image."""
        annotations = {}
        for img_path in self.image_files:
            # For each image, create a YOLO format annotation
            # Assuming single crater per image, centered
            # Format: <class> <x_center> <y_center> <width> <height>
            # All values normalized to [0, 1]
            annotations[img_path] = "0 0.5 0.5 0.3 0.3"  # Class 0, centered, 30% of image size
        return annotations
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Get annotation
        annotation = self.annotations[img_path]
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=np.array(image))
            image = transformed['image']
        
        # Convert annotation to tensor
        # Format: [class, x_center, y_center, width, height]
        annotation = torch.tensor([float(x) for x in annotation.split()])
        
        return image, annotation
    
    @staticmethod
    def get_default_transform():
        """Get default transform for images."""
        return A.Compose([
            A.Resize(640, 640),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    @staticmethod
    def get_train_transform():
        """Get transform with crater-specific augmentations for training."""
        return A.Compose([
            A.Resize(640, 640),
            # Crater-specific augmentations
            A.OneOf([
                A.GaussNoise(p=0.5),
                A.GaussianBlur(p=0.5),
                A.MotionBlur(p=0.5),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
                A.HueSaturationValue(p=0.5),
            ], p=0.3),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=1, p=0.5),
            ], p=0.3),
            # Standard augmentations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    @staticmethod
    def get_val_transform():
        """Get transform for validation."""
        return A.Compose([
            A.Resize(640, 640),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]) 