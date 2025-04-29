import os
import yaml
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

def load_config(config_path: str) -> dict:
    """Load visualization configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def visualize_detections(model_path: str, image_path: str, save_path: str, 
                        conf_threshold: float = 0.25, iou_threshold: float = 0.45):
    """Visualize detections from a single model on an image."""
    model = YOLO(model_path)
    results = model.predict(image_path, conf=conf_threshold, iou=iou_threshold)
    
    # Get the image with detections
    img = results[0].plot()
    
    # Save the visualization
    cv2.imwrite(save_path, img)
    return results[0]

def create_comparison_visualization(yolov5_results, yolov8_results, image_path: str, save_path: str):
    """Create a side-by-side comparison of YOLOv5 and YOLOv8 detections."""
    # Read original image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot YOLOv5 detections
    ax1.imshow(yolov5_results.plot())
    ax1.set_title('YOLOv5 Detections')
    ax1.axis('off')
    
    # Plot YOLOv8 detections
    ax2.imshow(yolov8_results.plot())
    ax2.set_title('YOLOv8 Detections')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def process_dataset(dataset_dir: str, output_dir: str, config: dict):
    """Process all images in a dataset directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of images
    image_files = [f for f in os.listdir(dataset_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    for img_file in image_files:
        img_path = os.path.join(dataset_dir, img_file)
        base_name = os.path.splitext(img_file)[0]
        
        # Create YOLOv5 visualization
        yolov5_save_path = os.path.join(output_dir, f'{base_name}_yolov5.jpg')
        yolov5_results = visualize_detections(
            'models/yolov5s.pt',
            img_path,
            yolov5_save_path,
            config['conf_threshold'],
            config['iou_threshold']
        )
        
        if config['compare_models']:
            # Create YOLOv8 visualization
            yolov8_save_path = os.path.join(output_dir, f'{base_name}_yolov8.jpg')
            yolov8_results = visualize_detections(
                'models/yolov8n.pt',
                img_path,
                yolov8_save_path,
                config['conf_threshold'],
                config['iou_threshold']
            )
            
            # Create comparison visualization
            comparison_path = os.path.join(output_dir, f'{base_name}_comparison.jpg')
            create_comparison_visualization(
                yolov5_results,
                yolov8_results,
                img_path,
                comparison_path
            )

def main():
    # Load configuration
    config = load_config('config.yaml')
    
    # Process each dataset
    datasets = ['train', 'valid', 'test']
    for dataset in datasets:
        dataset_dir = os.path.join('data', dataset, 'images')
        output_dir = os.path.join('results', 'visualizations', dataset)
        
        if os.path.exists(dataset_dir):
            print(f"\nProcessing {dataset} dataset...")
            process_dataset(dataset_dir, output_dir, config)
            print(f"Visualizations saved to: {output_dir}")
        else:
            print(f"Warning: {dataset_dir} not found, skipping...")

if __name__ == '__main__':
    main() 