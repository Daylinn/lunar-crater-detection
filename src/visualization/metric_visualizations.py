import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple
import cv2

def read_yolov5_detections(txt_path: str) -> List[Tuple[float, float, float, float, float]]:
    """Read detection results from a YOLO format txt file."""
    detections = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    confidence = float(parts[5]) if len(parts) > 5 else 1.0
                    detections.append((x_center, y_center, width, height, confidence))
    return detections

def read_yolov8_detections(img_path: str) -> List[Tuple[float, float, float, float, float]]:
    """Read detection results from a YOLOv8 output image by analyzing red bounding boxes."""
    detections = []
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Create a mask for red pixels
            red_mask = (img_rgb[:,:,0] > 200) & (img_rgb[:,:,1] < 50) & (img_rgb[:,:,2] < 50)
            red_mask = red_mask.astype(np.uint8) * 255
            
            # Find contours of red boxes
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            height, width = img.shape[:2]
            for contour in contours:
                # Filter out very small contours that might be noise
                if cv2.contourArea(contour) > 50:  # Reduced threshold to catch smaller boxes
                    x, y, w, h = cv2.boundingRect(contour)
                    # Convert to normalized coordinates
                    x_center = (x + w/2) / width
                    y_center = (y + h/2) / height
                    norm_w = w / width
                    norm_h = h / height
                    
                    # Calculate confidence based on the redness of the box
                    roi = img_rgb[y:y+h, x:x+w]
                    redness = np.mean(roi[:,:,0]) / 255.0
                    confidence = min(redness, 1.0)
                    
                    detections.append((x_center, y_center, norm_w, norm_h, confidence))
    return detections

def calculate_metrics(dataset_dir: str, yolo5_results: str, yolo8_results: str) -> Dict:
    """Calculate various metrics for both models' detections."""
    metrics = {
        'yolov5': {
            'total_detections': 0,
            'avg_confidence': 0,
            'size_distribution': [],
            'confidence_distribution': []
        },
        'yolov8': {
            'total_detections': 0,
            'avg_confidence': 0,
            'size_distribution': [],
            'confidence_distribution': []
        }
    }
    
    # Process YOLOv5 detections
    if os.path.exists(yolo5_results):
        for txt_file in os.listdir(yolo5_results):
            if txt_file.endswith('.txt'):
                txt_path = os.path.join(yolo5_results, txt_file)
                detections = read_yolov5_detections(txt_path)
                metrics['yolov5']['total_detections'] += len(detections)
                for det in detections:
                    _, _, width, height, conf = det
                    metrics['yolov5']['size_distribution'].append(width * height)
                    metrics['yolov5']['confidence_distribution'].append(conf)
    
    # Process YOLOv8 detections from result images
    if os.path.exists(yolo8_results):
        for img_file in os.listdir(yolo8_results):
            if img_file.endswith('.jpg'):  # YOLOv8 saves results as .jpg
                img_path = os.path.join(yolo8_results, img_file)
                detections = read_yolov8_detections(img_path)
                metrics['yolov8']['total_detections'] += len(detections)
                for det in detections:
                    _, _, width, height, conf = det
                    metrics['yolov8']['size_distribution'].append(width * height)
                    metrics['yolov8']['confidence_distribution'].append(conf)
    
    # Calculate averages
    for model in ['yolov5', 'yolov8']:
        if metrics[model]['total_detections'] > 0:
            metrics[model]['avg_confidence'] = np.mean(metrics[model]['confidence_distribution'])
    
    return metrics

def plot_detection_counts(metrics: Dict, output_dir: str):
    """Create bar chart comparing total detections."""
    plt.figure(figsize=(10, 6))
    models = ['YOLOv5', 'YOLOv8']
    counts = [metrics['yolov5']['total_detections'], metrics['yolov8']['total_detections']]
    
    plt.bar(models, counts, color=['green', 'red'])
    plt.title('Total Detections Comparison')
    plt.ylabel('Number of Detections')
    plt.savefig(os.path.join(output_dir, 'detection_counts.png'))
    plt.close()

def plot_confidence_distributions(metrics: Dict, output_dir: str):
    """Create histogram of confidence scores."""
    plt.figure(figsize=(12, 6))
    
    plt.hist(metrics['yolov5']['confidence_distribution'], bins=20, alpha=0.5, color='green', label='YOLOv5')
    plt.hist(metrics['yolov8']['confidence_distribution'], bins=20, alpha=0.5, color='red', label='YOLOv8')
    
    plt.title('Confidence Score Distribution')
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'))
    plt.close()

def plot_size_distributions(metrics: Dict, output_dir: str):
    """Create box plot of detection sizes."""
    plt.figure(figsize=(10, 6))
    
    data = [metrics['yolov5']['size_distribution'], metrics['yolov8']['size_distribution']]
    plt.boxplot(data, labels=['YOLOv5', 'YOLOv8'], patch_artist=True,
                boxprops=dict(facecolor='lightgreen', color='green'),
                whiskerprops=dict(color='green'),
                capprops=dict(color='green'),
                medianprops=dict(color='darkgreen'))
    
    plt.title('Detection Size Distribution')
    plt.ylabel('Normalized Area')
    plt.savefig(os.path.join(output_dir, 'size_distribution.png'))
    plt.close()

def plot_performance_comparison(metrics: Dict, output_dir: str):
    """Create radar chart comparing multiple metrics."""
    metrics_names = ['Total Detections', 'Average Confidence', 'Median Size']
    yolo5_values = [
        metrics['yolov5']['total_detections'],
        metrics['yolov5']['avg_confidence'],
        np.median(metrics['yolov5']['size_distribution']) if metrics['yolov5']['size_distribution'] else 0
    ]
    yolo8_values = [
        metrics['yolov8']['total_detections'],
        metrics['yolov8']['avg_confidence'],
        np.median(metrics['yolov8']['size_distribution']) if metrics['yolov8']['size_distribution'] else 0
    ]
    
    angles = np.linspace(0, 2*np.pi, len(metrics_names), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    
    yolo5_values = np.concatenate((yolo5_values, [yolo5_values[0]]))
    yolo8_values = np.concatenate((yolo8_values, [yolo8_values[0]]))
    
    ax.plot(angles, yolo5_values, 'o-', color='green', label='YOLOv5')
    ax.plot(angles, yolo8_values, 'o-', color='red', label='YOLOv8')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_names)
    plt.title('Model Performance Comparison')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'))
    plt.close()

def main():
    # Create output directory
    output_dir = 'metric_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # Define paths
    yolo5_results = 'results/yolov5/labels'
    yolo8_results = 'results/yolov8/detection_results_20250410_170653/results'
    
    # Calculate metrics for each dataset
    for dataset in ['train', 'test', 'valid']:
        dataset_dir = os.path.join('data', dataset, 'images')
        dataset_output_dir = os.path.join(output_dir, dataset)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # Calculate metrics
        metrics = calculate_metrics(dataset_dir, yolo5_results, yolo8_results)
        
        # Generate visualizations
        plot_detection_counts(metrics, dataset_output_dir)
        plot_confidence_distributions(metrics, dataset_output_dir)
        plot_size_distributions(metrics, dataset_output_dir)
        plot_performance_comparison(metrics, dataset_output_dir)
        
        # Save metrics to JSON
        with open(os.path.join(dataset_output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

if __name__ == '__main__':
    main() 