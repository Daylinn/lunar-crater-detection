import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2

class LunarDataProcessor:
    def __init__(self, 
                 image_size: int = 640,
                 augment: bool = True,
                 normalize: bool = True):
        """
        Initialize the data processor with augmentation and preprocessing settings.
        
        Args:
            image_size: Target size for resizing images
            augment: Whether to apply data augmentation
            normalize: Whether to normalize pixel values
        """
        self.image_size = image_size
        self.augment = augment
        self.normalize = normalize
        
        # Define augmentation pipeline
        self.augmentation = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5),
            A.GaussNoise(p=0.3),
            A.GaussianBlur(p=0.3),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if normalize else A.ToFloat(),
            ToTensorV2()
        ])
        
        # Define validation/test preprocessing pipeline
        self.preprocess = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if normalize else A.ToFloat(),
            ToTensorV2()
        ])
    
    def process_image(self, 
                     image_path: str, 
                     is_training: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Process a single image with augmentation for training or preprocessing for validation/test.
        
        Args:
            image_path: Path to the image file
            is_training: Whether the image is for training (affects augmentation)
            
        Returns:
            Tuple of (processed_image, augmentation_info)
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply augmentation or preprocessing
        if is_training and self.augment:
            augmented = self.augmentation(image=image)
            processed_image = augmented['image']
            augmentation_info = {
                'brightness_contrast': augmented.get('brightness_contrast', None),
                'gamma': augmented.get('gamma', None),
                'rotation': augmented.get('rotation', None),
                'flip': augmented.get('flip', None)
            }
        else:
            processed = self.preprocess(image=image)
            processed_image = processed['image']
            augmentation_info = {}
        
        return processed_image, augmentation_info
    
    def process_batch(self, 
                     image_paths: List[str], 
                     is_training: bool = True) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Process a batch of images.
        
        Args:
            image_paths: List of paths to image files
            is_training: Whether the images are for training
            
        Returns:
            Tuple of (processed_images, augmentation_infos)
        """
        processed_images = []
        augmentation_infos = []
        
        for image_path in image_paths:
            processed_image, augmentation_info = self.process_image(image_path, is_training)
            processed_images.append(processed_image)
            augmentation_infos.append(augmentation_info)
        
        return processed_images, augmentation_infos
    
    def create_data_yaml(self, 
                        data_dir: str, 
                        train_ratio: float = 0.7, 
                        val_ratio: float = 0.2) -> str:
        """
        Create YAML configuration file for YOLOv8 training.
        
        Args:
            data_dir: Root directory containing the dataset
            train_ratio: Ratio of data to use for training
            val_ratio: Ratio of data to use for validation
            
        Returns:
            Path to the created YAML file
        """
        data_dir = Path(data_dir)
        images_dir = data_dir / 'images'
        labels_dir = data_dir / 'labels'
        
        # Create train/val/test splits
        image_files = sorted([f for f in images_dir.glob('*.jpg')])
        n_images = len(image_files)
        
        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)
        
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train+n_val]
        test_files = image_files[n_train+n_val:]
        
        # Create directories
        splits = ['train', 'val', 'test']
        for split in splits:
            (data_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (data_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Move files to appropriate directories
        for files, split in zip([train_files, val_files, test_files], splits):
            for img_path in files:
                # Move image
                new_img_path = data_dir / split / 'images' / img_path.name
                img_path.rename(new_img_path)
                
                # Move corresponding label
                label_path = labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    new_label_path = data_dir / split / 'labels' / label_path.name
                    label_path.rename(new_label_path)
        
        # Create YAML file
        yaml_content = f"""path: {data_dir.absolute()}  # dataset root dir
train: train/images  # train images
val: val/images  # val images
test: test/images  # test images

# Classes
names:
  0: crater
"""
        
        yaml_path = data_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        return str(yaml_path)

def main():
    processor = LunarDataProcessor()
    processor.create_data_yaml()

if __name__ == "__main__":
    main() 