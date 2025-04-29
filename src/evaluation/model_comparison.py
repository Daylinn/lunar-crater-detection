import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import cv2
from collections import defaultdict
from pathlib import Path
import pandas as pd

def read_detections(file_path):
    """Read detection results from a YOLO format text file."""
    detections = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                values = line.strip().split()
                if len(values) >= 6:  # class x y w h conf
                    detections.append({
                        'class': int(values[0]),
                        'x': float(values[1]),
                        'y': float(values[2]),
                        'w': float(values[3]),
                        'h': float(values[4]),
                        'conf': float(values[5])
                    })
                elif len(values) >= 5:  # class x y w h
                    detections.append({
                        'class': int(values[0]),
                        'x': float(values[1]),
                        'y': float(values[2]),
                        'w': float(values[3]),
                        'h': float(values[4]),
                        'conf': 1.0  # Default confidence if not provided
                    })
    return detections

def analyze_model_results(results_dir):
    """Analyze detection results for a model."""
    total_detections = 0
    confidence_scores = []
    box_sizes = []
    num_images = 0
    
    # Process all txt files in the directory
    for file in os.listdir(results_dir):
        if file.endswith('.txt'):
            num_images += 1
            detections = read_detections(os.path.join(results_dir, file))
            total_detections += len(detections)
            
            for det in detections:
                confidence_scores.append(det['conf'])
                box_sizes.append(det['w'] * det['h'])  # normalized area
    
    # Calculate metrics
    metrics = {
        'total_detections': total_detections,
        'avg_detections_per_image': total_detections / max(1, num_images),
        'confidence_scores': confidence_scores,
        'box_sizes': box_sizes,
        'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
        'confidence_std': np.std(confidence_scores) if confidence_scores else 0,
        'avg_box_size': np.mean(box_sizes) if box_sizes else 0,
        'box_size_std': np.std(box_sizes) if box_sizes else 0
    }
    
    return metrics

def plot_comparison(yolov5_metrics, yolov8_metrics):
    """Generate comparison plots between YOLOv5 and YOLOv8."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_dir = Path('comparison_plots')
    plot_dir.mkdir(exist_ok=True)
    
    # 1. Confidence Score Distribution
    plt.figure(figsize=(10, 6))
    plt.title('Confidence Score Distribution')
    sns.kdeplot(data=[yolov5_metrics['confidence_scores'], yolov8_metrics['confidence_scores']], 
                label=['YOLOv5', 'YOLOv8'])
    plt.xlabel('Confidence Score')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(plot_dir / f'confidence_distribution_{timestamp}.png')
    plt.close()
    
    # 2. Box Size Distribution
    plt.figure(figsize=(10, 6))
    plt.title('Bounding Box Size Distribution')
    sns.kdeplot(data=[yolov5_metrics['box_sizes'], yolov8_metrics['box_sizes']], 
                label=['YOLOv5', 'YOLOv8'])
    plt.xlabel('Box Size (normalized)')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(plot_dir / f'box_size_distribution_{timestamp}.png')
    plt.close()
    
    # 3. Detection Count Comparison
    plt.figure(figsize=(8, 6))
    plt.title('Total Detections Comparison')
    plt.bar(['YOLOv5', 'YOLOv8'], 
            [yolov5_metrics['total_detections'], yolov8_metrics['total_detections']])
    plt.ylabel('Number of Detections')
    plt.savefig(plot_dir / f'detection_count_{timestamp}.png')
    plt.close()
    
    # 4. Average Metrics Comparison
    metrics = ['avg_confidence', 'avg_detections_per_image']
    values_yolov5 = [yolov5_metrics['avg_confidence'], yolov5_metrics['avg_detections_per_image']]
    values_yolov8 = [yolov8_metrics['avg_confidence'], yolov8_metrics['avg_detections_per_image']]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, values_yolov5, width, label='YOLOv5')
    plt.bar(x + width/2, values_yolov8, width, label='YOLOv8')
    
    plt.title('Average Metrics Comparison')
    plt.xticks(x, ['Avg Confidence', 'Avg Detections/Image'])
    plt.legend()
    plt.savefig(plot_dir / f'average_metrics_{timestamp}.png')
    plt.close()

def main():
    # Define paths for YOLOv5 and YOLOv8 results
    yolov5_results_dir = "data/test/labels"  # YOLOv5 results are in the test labels directory
    yolov8_results_dir = "data/test/images"  # YOLOv8 results are alongside the test images
    
    # Analyze results
    yolov5_metrics = analyze_model_results(yolov5_results_dir)
    yolov8_metrics = analyze_model_results(yolov8_results_dir)
    
    # Print comparison
    print("\n=== Model Performance Comparison ===\n")
    print("YOLOv5 Metrics:")
    print(f"Total Detections: {yolov5_metrics['total_detections']}")
    print(f"Average Detections per Image: {yolov5_metrics['avg_detections_per_image']:.2f}")
    print(f"Average Confidence: {yolov5_metrics['avg_confidence']:.3f} (±{yolov5_metrics['confidence_std']:.3f})")
    print(f"Average Box Size: {yolov5_metrics['avg_box_size']:.3f} (±{yolov5_metrics['box_size_std']:.3f})\n")
    
    print("YOLOv8 Metrics:")
    print(f"Total Detections: {yolov8_metrics['total_detections']}")
    print(f"Average Detections per Image: {yolov8_metrics['avg_detections_per_image']:.2f}")
    print(f"Average Confidence: {yolov8_metrics['avg_confidence']:.3f} (±{yolov8_metrics['confidence_std']:.3f})")
    print(f"Average Box Size: {yolov8_metrics['avg_box_size']:.3f} (±{yolov8_metrics['box_size_std']:.3f})\n")
    
    # Plot comparison
    plot_comparison(yolov5_metrics, yolov8_metrics)
    print("Detailed comparison plots and metrics have been saved in the 'comparison_plots' directory.")

if __name__ == '__main__':
    main() 