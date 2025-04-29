import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple

def read_yolov5_detections(txt_path):
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
                    detections.append((class_id, x_center, y_center, width, height, confidence))
    return detections

def get_image_differences(img1, img2):
    diff = cv2.absdiff(img1, img2)
    return diff

def draw_detections(image, detections, color, thickness=2):
    h, w = image.shape[:2]
    for det in detections:
        class_id, x_center, y_center, width, height, confidence = det
        x1 = int((x_center - width/2) * w)
        y1 = int((y_center - height/2) * h)
        x2 = int((x_center + width/2) * w)
        y2 = int((y_center + height/2) * h)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        label = f'Crater {confidence:.2f}'
        cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

def process_dataset(dataset_name):
    print(f"\nProcessing {dataset_name} dataset...")
    dataset_dir = os.path.join("data", dataset_name, "images")
    output_dir = os.path.join("comparison_visualizations", dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files in the dataset directory
    image_files = [f for f in os.listdir(dataset_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    for img_file in image_files:
        print(f"Processing {dataset_name}/{img_file}")
        
        # Read original image
        orig_img_path = os.path.join(dataset_dir, img_file)
        orig_img = cv2.imread(orig_img_path)
        if orig_img is None:
            print(f"Could not read original image: {orig_img_path}")
            continue
            
        # Read YOLOv5 detections
        yolo5_txt_path = os.path.join("results/yolov5/detection_results_20250410_170653/labels", 
                                     os.path.splitext(img_file)[0] + ".txt")
        yolo5_dets = read_yolov5_detections(yolo5_txt_path)
        
        # Read YOLOv8 detections
        yolo8_txt_path = os.path.join("results/yolov8/detection_results_20250410_170653/results", 
                                     os.path.splitext(img_file)[0] + ".txt")
        yolo8_dets = read_yolov5_detections(yolo8_txt_path)
        
        # Create comparison visualization
        comparison = orig_img.copy()
        draw_detections(comparison, yolo5_dets, (0, 255, 0))  # Green for YOLOv5
        draw_detections(comparison, yolo8_dets, (0, 0, 255))  # Red for YOLOv8
        
        # Save comparison
        output_path = os.path.join(output_dir, f"comparison_{img_file}")
        cv2.imwrite(output_path, comparison)
        print(f"Saved comparison to {output_path}")

def main():
    # Create output directory
    os.makedirs("comparison_visualizations", exist_ok=True)
    
    # Process each dataset
    for dataset in ["train", "test", "valid"]:
        process_dataset(dataset)

if __name__ == "__main__":
    main() 