"""
Evaluation script for comparing YOLOv5 and YOLOv8 performance on lunar crater detection.
This version explicitly finds images and processes them one by one.
"""

import os
import yaml
import torch
import numpy as np
from typing import Dict, List, Tuple
import time
from ultralytics import YOLO
import glob # To find images
import cv2 # Needed if we want image dimensions, but not strictly necessary for this version

# --- Configuration Loading ---
def load_config(config_path: str) -> Dict:
    """Loads the YAML config file with model paths and parameters."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"ERROR: Config file not found at {config_path}")
        return None
    except Exception as e:
        print(f"ERROR reading config file {config_path}: {e}")
        return None

# --- Model Loading ---
# Note: Duplicated code from ensemble script. Refactor candidate!
def load_yolov5_model(model_path: str):
    """Loads a YOLOv5 model."""
    try:
        # Using force_reload=True based on previous error
        # Temporarily loading standard 'yolov5s' instead of 'custom' for testing
        print(f"Attempting to load standard 'yolov5s' model (ignoring path: {model_path})...")
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True) 
        # model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True) # Original line
        model.conf = 0.25  # Use consistent thresholds
        model.iou = 0.45
        print(f"Successfully loaded standard YOLOv5s model.") # Modified success message
        return model
    except Exception as e:
        print(f"ERROR loading YOLOv5 model {model_path}: {str(e)}") # Keep original path in error
        return None

def load_yolov8_model(model_path: str):
    """Loads a YOLOv8 model."""
    try:
        model = YOLO(model_path)
        print(f"Successfully loaded YOLOv8 model from {model_path}")
        return model
    except Exception as e:
        print(f"ERROR loading YOLOv8 model {model_path}: {str(e)}")
        return None

# --- Process Single Image (Helper) ---
# Refactored to process one image at a time

def process_image_yolov5(model, image_path: str, img_size: int) -> Dict:
    """Runs YOLOv5 inference on a single image and calculates metrics."""
    try:
        start_time = time.time()
        # Run inference on the single image path
        results = model(image_path, size=img_size)
        inference_time = time.time() - start_time

        # Extract results
        preds = results.pandas().xyxy[0]
        num_detections = len(preds)
        avg_confidence = preds['confidence'].mean() if num_detections > 0 else 0

        # Calculate average diameter in pixels
        avg_diameter_pixels = 0
        if num_detections > 0:
            widths = preds['xmax'] - preds['xmin']
            heights = preds['ymax'] - preds['ymin']
            diameters = (widths + heights) / 2
            avg_diameter_pixels = diameters.mean()

        metrics = {
            'inference_time': inference_time,
            'num_detections': num_detections,
            'average_confidence': avg_confidence,
            'average_diameter_pixels': avg_diameter_pixels,
            'processed': True
        }
        return metrics

    except Exception as e:
        print(f"  ERROR processing {os.path.basename(image_path)} with YOLOv5: {e}")
        return {'inference_time': 0, 'num_detections': 0, 'average_confidence': 0, 'average_diameter_pixels': 0, 'processed': False}

def process_image_yolov8(model, image_path: str, img_size: int) -> Dict:
    """Runs YOLOv8 inference on a single image and calculates metrics."""
    try:
        start_time = time.time()
        # Run inference on the single image path
        results = model(image_path, imgsz=img_size, conf=0.25, iou=0.45, verbose=False)
        inference_time = time.time() - start_time

        # Extract results
        num_detections = 0
        avg_confidence = 0
        avg_diameter_pixels = 0
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            num_detections = len(boxes)
            avg_confidence = boxes.conf.mean().item()

            # Calculate average diameter
            xyxy = boxes.xyxy.cpu().numpy()
            widths = xyxy[:, 2] - xyxy[:, 0]
            heights = xyxy[:, 3] - xyxy[:, 1]
            diameters = (widths + heights) / 2
            avg_diameter_pixels = np.mean(diameters)

        metrics = {
            'inference_time': inference_time,
            'num_detections': num_detections,
            'average_confidence': avg_confidence,
            'average_diameter_pixels': avg_diameter_pixels,
            'processed': True
        }
        return metrics

    except Exception as e:
        print(f"  ERROR processing {os.path.basename(image_path)} with YOLOv8: {e}")
        return {'inference_time': 0, 'num_detections': 0, 'average_confidence': 0, 'average_diameter_pixels': 0, 'processed': False}

# --- Main Execution ---
def main():
    """Main function to run the evaluation pipeline."""
    config_path = 'config/evaluation.yaml'
    print(f"Loading configuration from {config_path}...")
    config = load_config(config_path)
    if config is None: return

    # --- Load Models ---
    print("\nLoading models...")
    yolov5_model = load_yolov5_model(config['yolov5_model_path'])
    yolov8_model = load_yolov8_model(config['yolov8_model_path'])

    # Exit if models didn't load
    if not yolov5_model or not yolov8_model:
        print("\nFailed to load one or both models. Exiting.")
        return
    print("Models loaded successfully!")

    # --- Find Images (Explicitly) ---
    print("\nFinding evaluation images...")
    image_paths = find_evaluation_images(config['data_yaml'])

    # Exit if no images found
    if not image_paths:
        print("\nNo images found for evaluation. Exiting.")
        return

    # --- Run Evaluation Loop ---
    img_size = config.get('img_size', 640)
    all_metrics_v5 = []
    all_metrics_v8 = []

    print(f"\nStarting evaluation on {len(image_paths)} images (img_size={img_size})...")
    image_count = 0
    for image_path in image_paths:
        image_count += 1
        print(f"-- Processing image {image_count}/{len(image_paths)}: {os.path.basename(image_path)} --")

        # Process with YOLOv5
        metrics_v5 = process_image_yolov5(yolov5_model, image_path, img_size)
        all_metrics_v5.append(metrics_v5)

        # Process with YOLOv8
        metrics_v8 = process_image_yolov8(yolov8_model, image_path, img_size)
        all_metrics_v8.append(metrics_v8)

    # --- Aggregate and Save Results ---
    print("\nAggregating results...")
    final_metrics = {}

    # Filter out results for images that failed processing
    valid_metrics_v5 = [m for m in all_metrics_v5 if m['processed']]
    valid_metrics_v8 = [m for m in all_metrics_v8 if m['processed']]

    for name, metrics_list in zip(['yolov5', 'yolov8'], [valid_metrics_v5, valid_metrics_v8]):
        if metrics_list:
            valid_metrics_for_conf = [m['average_confidence'] for m in metrics_list if m['num_detections'] > 0]
            valid_metrics_for_diam = [m['average_diameter_pixels'] for m in metrics_list if m['num_detections'] > 0]

            final_metrics[name] = {
                'average_inference_time_s': np.mean([m['inference_time'] for m in metrics_list]) if metrics_list else 0,
                'average_num_detections': np.mean([m['num_detections'] for m in metrics_list]) if metrics_list else 0,
                'overall_average_confidence': np.mean(valid_metrics_for_conf) if valid_metrics_for_conf else 0,
                'overall_average_diameter_pixels': np.mean(valid_metrics_for_diam) if valid_metrics_for_diam else 0,
                'images_processed': len(metrics_list),
                'images_failed': len(all_metrics_v5 if name == 'yolov5' else all_metrics_v8) - len(metrics_list)
            }
            print(f"  Aggregated metrics for {name}: {final_metrics[name]}")
        else:
             print(f"WARN: No metrics collected successfully for {name}.")
             final_metrics[name] = {'error': 'No images processed successfully',
                                    'images_processed': 0,
                                    'images_failed': len(all_metrics_v5 if name == 'yolov5' else all_metrics_v8)}

    # --- Save Results ---
    output_dir = 'results/evaluation'
    output_path = os.path.join(output_dir, 'metrics.json')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving final aggregated metrics to {output_path}...")
    try:
        with open(output_path, 'w') as f:
            yaml.dump(final_metrics, f, indent=4, default_flow_style=False)
        print("Evaluation complete. Results saved.")
    except Exception as e:
        print(f"ERROR saving results: {e}")

def find_evaluation_images(data_yaml_path: str) -> List[str]:
    """Finds images for evaluation based on data.yaml configuration."""
    try:
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Get the base path from data.yaml
        base_path = data_config.get('path', '')
        if not base_path:
            print("WARNING: No 'path' specified in data.yaml, using current directory")
            base_path = '.'
        
        # Get validation path, defaulting to 'valid/images' if not specified
        val_path = data_config.get('val', 'valid/images')
        
        # Construct full path relative to data.yaml location
        full_path = os.path.join(base_path, val_path)
        print(f"Searching for images in: {full_path}")
        
        # Find all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(full_path, ext)))
        
        if not image_files:
            print(f"ERROR: No images found in {full_path}")
            return []
            
        print(f"Found {len(image_files)} images for evaluation")
        return image_files
    except Exception as e:
        print(f"ERROR finding evaluation images: {str(e)}")
        return []

if __name__ == "__main__":
    main()