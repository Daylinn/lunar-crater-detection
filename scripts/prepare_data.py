import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

def create_directories():
    """Create necessary directories for YOLO training"""
    base_dirs = ['train', 'valid', 'test']
    sub_dirs = ['images', 'labels']
    
    for base_dir in base_dirs:
        for sub_dir in sub_dirs:
            Path(f'../data/{base_dir}/{sub_dir}').mkdir(parents=True, exist_ok=True)

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

def split_dataset(source_dir, train_ratio=0.8, val_ratio=0.1):
    """Split dataset into train/val/test sets"""
    # Get all image files
    image_files = list(Path(source_dir).glob('*.jpg')) + list(Path(source_dir).glob('*.png'))
    np.random.shuffle(image_files)
    
    # Calculate split indices
    n_total = len(image_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split files
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train+n_val]
    test_files = image_files[n_train+n_val:]
    
    # Move files to appropriate directories
    for img_path in train_files:
        label_path = img_path.with_suffix('.txt')
        shutil.copy2(img_path, f'../data/train/images/{img_path.name}')
        if label_path.exists():
            shutil.copy2(label_path, f'../data/train/labels/{label_path.name}')
    
    for img_path in val_files:
        label_path = img_path.with_suffix('.txt')
        shutil.copy2(img_path, f'../data/valid/images/{img_path.name}')
        if label_path.exists():
            shutil.copy2(label_path, f'../data/valid/labels/{label_path.name}')
    
    for img_path in test_files:
        label_path = img_path.with_suffix('.txt')
        shutil.copy2(img_path, f'../data/test/images/{img_path.name}')
        if label_path.exists():
            shutil.copy2(label_path, f'../data/test/labels/{label_path.name}')

if __name__ == "__main__":
    # Create directories
    create_directories()
    
    # Analyze dataset
    print("Analyzing dataset...")
    analyze_dataset()
    
    # Visualize a sample
    print("\nVisualizing a sample image...")
    sample_img = next(Path('../data/train/images').glob('*'))
    sample_label = Path('../data/train/labels') / sample_img.name.replace('.jpg', '.txt')
    visualize_sample(sample_img, sample_label) 