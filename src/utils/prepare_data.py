"""
My lunar crater dataset preparation script.
This script helps us organize our lunar images and labels into the right format for training.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import random

def create_directories():
    """Create the directory structure we need for our dataset."""
    # Set up our main data directory
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Create directories for our dataset splits
    for split in ['train', 'valid', 'test']:
        (data_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (data_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

def visualize_sample(image_path, label_path):
    """Visualize a sample image with its bounding boxes"""
    # Read image
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    
    # Create figure
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    
    # Read and plot bounding boxes
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                class_id, x, y, w, h = map(float, line.strip().split())
                
                # Convert YOLO format to pixel coordinates
                img_h, img_w = img.shape[:2]
                x1 = int((x - w/2) * img_w)
                y1 = int((y - h/2) * img_h)
                x2 = int((x + w/2) * img_w)
                y2 = int((y + h/2) * img_h)
                
                # Draw rectangle
                plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                                fill=False, color='red', linewidth=2))
    
    plt.axis('off')
    plt.show()

def analyze_dataset():
    """Analyze the dataset statistics"""
    data_dir = Path('../data')
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        images_dir = data_dir / split / 'images'
        labels_dir = data_dir / split / 'labels'
        
        n_images = len(list(images_dir.glob('*')))
        n_labels = len(list(labels_dir.glob('*')))
        
        print(f"\n{split.upper()} Set Statistics:")
        print(f"Number of images: {n_images}")
        print(f"Number of labels: {n_labels}")
        
        if n_images > 0:
            # Analyze image sizes
            img_sizes = []
            for img_path in images_dir.glob('*'):
                img = cv2.imread(str(img_path))
                if img is not None:
                    img_sizes.append(img.shape[:2])
            
            if img_sizes:
                img_sizes = np.array(img_sizes)
                print(f"Average image size: {img_sizes.mean(axis=0).astype(int)}")
                print(f"Min image size: {img_sizes.min(axis=0)}")
                print(f"Max image size: {img_sizes.max(axis=0)}")
        
        # Analyze labels
        if n_labels > 0:
            n_craters = 0
            for label_path in labels_dir.glob('*'):
                with open(label_path, 'r') as f:
                    n_craters += sum(1 for line in f)
            
            print(f"Total number of craters: {n_craters}")
            print(f"Average craters per image: {n_craters/n_images:.2f}")

def split_dataset(image_dir, label_dir, train_ratio=0.8, valid_ratio=0.1):
    """
    Split our dataset into training, validation, and test sets.
    We'll use 80% for training, 10% for validation, and 10% for testing.
    """
    # Get all our image files
    image_files = list(Path(image_dir).glob('*.jpg'))
    random.shuffle(image_files)
    
    # Calculate how many images go in each split
    total_images = len(image_files)
    train_size = int(total_images * train_ratio)
    valid_size = int(total_images * valid_ratio)
    
    # Split our files
    train_files = image_files[:train_size]
    valid_files = image_files[train_size:train_size + valid_size]
    test_files = image_files[train_size + valid_size:]
    
    # Move files to their respective directories
    for files, split in [(train_files, 'train'), (valid_files, 'valid'), (test_files, 'test')]:
        for img_path in files:
            # Get the corresponding label file
            label_path = Path(label_dir) / f"{img_path.stem}.txt"
            
            # Move image and label to their new locations
            shutil.copy2(img_path, f'data/{split}/images/{img_path.name}')
            if label_path.exists():
                shutil.copy2(label_path, f'data/{split}/labels/{label_path.name}')

def create_yaml():
    """Create the YAML configuration file for our dataset."""
    yaml_content = """
path: data  # dataset root dir
train: train/images  # train images (relative to 'path')
val: valid/images  # val images (relative to 'path')
test: test/images  # test images (relative to 'path')

# Classes
names:
  0: crater  # crater class
"""
    
    with open('data/lunar_craters.yaml', 'w') as f:
        f.write(yaml_content)

def main():
    """Main function to prepare our dataset."""
    # Create our directory structure
    create_directories()
    
    # Split our dataset
    split_dataset('raw_data/images', 'raw_data/labels')
    
    # Create our YAML config
    create_yaml()
    
    print("Dataset preparation complete! Check the 'data' directory for the organized dataset.")

if __name__ == '__main__':
    main() 