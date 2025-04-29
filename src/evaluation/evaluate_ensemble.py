"""
Evaluation script for comparing YOLOv5, YOLOv8, and an Ensemble model 
using Weighted Boxes Fusion (WBF) for lunar crater detection.
"""

import os
import yaml
import torch
import numpy as np
from typing import Dict, List, Tuple
import time
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
import glob # To find images
import cv2 # To load images for dimensions

# --- Configuration Loading ---
def load_config(config_path: str) -> Dict:
    """Loads the YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# --- Model Loading Utilities ---

def load_yolov5_model(model_path: str):
    """Loads a YOLOv5 model."""
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, verbose=False)
        model.conf = 0.25  # Use consistent thresholds
        model.iou = 0.45
        return model
    except Exception as e:
        print(f"Error loading YOLOv5 model {model_path}: {str(e)}")
        return None

def load_yolov8_model(model_path: str):
    """Loads a YOLOv8 model."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"Error loading YOLOv8 model {model_path}: {str(e)}")
        return None

# --- Inference and Prediction Formatting ---

def predict_yolov5(model, image_path: str, img_size: int) -> Tuple[List, List, List]:
    """Runs YOLOv5 inference and formats predictions for WBF."""
    results = model(image_path, size=img_size)
    preds = results.pandas().xyxy[0]
    boxes = (preds[['xmin', 'ymin', 'xmax', 'ymax']]).values.tolist()
    scores = preds['confidence'].values.tolist()
    labels = preds['class'].values.tolist() # Assuming class 0 for craters
    # Normalize boxes 0-1 range based on image size used for inference
    img_h, img_w = img_size, img_size # Approximation, ideally read actual image size
    boxes_normalized = []
    if len(boxes) > 0: # Check if boxes list is not empty
        boxes_normalized = [[x1/img_w, y1/img_h, x2/img_w, y2/img_h] for x1, y1, x2, y2 in boxes]
    return boxes_normalized, scores, labels

def predict_yolov8(model, image_path: str, img_size: int) -> Tuple[List, List, List]:
    """Runs YOLOv8 inference and formats predictions for WBF."""
    results = model(image_path, imgsz=img_size, conf=0.25, iou=0.45, verbose=False) # Use consistent thresholds
    boxes_normalized = []
    scores = []
    labels = []
    if len(results) > 0 and len(results[0].boxes) > 0:
        boxes_normalized = results[0].boxes.xyxyn.cpu().numpy().tolist()
        scores = results[0].boxes.conf.cpu().numpy().tolist()
        labels = results[0].boxes.cls.cpu().numpy().tolist()
    return boxes_normalized, scores, labels

# --- Ensemble Function ---

def run_wbf(boxes_list: List, scores_list: List, labels_list: List, 
            image_dims: Tuple[int, int], iou_thr: float = 0.5, skip_box_thr: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Applies Weighted Boxes Fusion."""
    img_h, img_w = image_dims
    # WBF expects boxes in [x1, y1, x2, y2] format, normalized 0-1.
    # Labels should be integers.
    
    # Ensure labels are integers
    int_labels_list = []
    for labels in labels_list:
        try:
            int_labels_list.append([int(l) for l in labels])
        except ValueError as e:
             print(f"Warning: Could not convert labels to integers: {labels}. Skipping this list for WBF.")
             int_labels_list.append([]) # Append empty list if conversion fails for any label in the list

    # Filter out empty lists to avoid errors in WBF if a model had no predictions or failed label conversion
    valid_indices = [i for i, boxes in enumerate(boxes_list) if boxes and int_labels_list[i]]
    if not valid_indices:
        print("Warning: No valid predictions to fuse after filtering. Returning empty results.")
        return np.array([]), np.array([]), np.array([])
    
    filtered_boxes = [boxes_list[i] for i in valid_indices]
    filtered_scores = [scores_list[i] for i in valid_indices]
    filtered_labels = [int_labels_list[i] for i in valid_indices]

    
    try:
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            filtered_boxes,
            filtered_scores,
            filtered_labels,
            weights=None, # Default: equal weights for each model
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr
        )
    except Exception as e:
        print(f"Error during WBF execution: {e}")
        return np.array([]), np.array([]), np.array([])

    
    # Denormalize boxes back to image coordinates
    fused_boxes_denormalized = fused_boxes * np.array([img_w, img_h, img_w, img_h])
    return fused_boxes_denormalized, fused_scores, fused_labels

# --- Metrics Calculation ---

def calculate_metrics(predictions: Tuple[np.ndarray, np.ndarray, np.ndarray], inference_time: float) -> Dict:
    """Calculates metrics from predictions."""
    boxes, scores, _ = predictions
    num_detections = len(boxes)
    avg_confidence = scores.mean() if num_detections > 0 else 0
    
    # Calculate average diameter in pixels
    if num_detections > 0:
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        diameters = (widths + heights) / 2
        avg_diameter_pixels = diameters.mean()
    else:
        avg_diameter_pixels = 0
        
    return {
        'inference_time': inference_time,
        'num_detections': num_detections,
        'average_confidence': avg_confidence,
        'average_diameter_pixels': avg_diameter_pixels
    }

# --- Main Execution ---

def main():
    config_path = 'config/evaluation.yaml'
    config = load_config(config_path)
    if config is None:
        print("Failed to load configuration. Exiting.")
        return
    
    # Load models
    print("Loading models...")
    yolov5_model = load_yolov5_model(config['yolov5_model_path'])
    yolov8_model = load_yolov8_model(config['yolov8_model_path'])
    
    if not yolov5_model or not yolov8_model:
        print("Failed to load one or both models. Exiting.")
        return

    # Find images (assuming they are in a directory specified in data.yaml)
    try:
        with open(config['data_yaml'], 'r') as f:
            data_config = yaml.safe_load(f)
        # Look for validation images path, fallback to train
        image_dir_key = 'val' if 'val' in data_config else 'train'
        if image_dir_key not in data_config:
             print(f"Error: Could not find '{image_dir_key}' image directory path in {config['data_yaml']}")
             return
        # Correct path relative to data.yaml location
        data_yaml_dir = os.path.dirname(config['data_yaml'])
        image_dir = os.path.join(data_yaml_dir, data_config[image_dir_key])
        image_paths = glob.glob(os.path.join(image_dir, '*.jpg')) + \
                      glob.glob(os.path.join(image_dir, '*.png')) + \
                      glob.glob(os.path.join(image_dir, '*.tif'))
        if not image_paths:
            print(f"Error: No images found in {image_dir}")
            return
        print(f"Found {len(image_paths)} images for evaluation in {image_dir}")
    except Exception as e:
        print(f"Error processing data yaml or finding images: {e}")
        return
        
    img_size = config.get('img_size', 640) # Default to 640 if not specified
    iou_thr_wbf = config.get('wbf_iou_thr', 0.55) # Default WBF IoU threshold
    skip_box_thr_wbf = config.get('wbf_skip_box_thr', 0.1) # Default WBF confidence threshold

    all_metrics_v5 = []
    all_metrics_v8 = []
    all_metrics_ensemble = []

    print(f"Evaluating {len(image_paths)} images...")
    for image_path in image_paths:
        try:
            # Get actual image dimensions for denormalization
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not read image {image_path}. Skipping.")
                continue
            img_h, img_w, _ = img.shape
            image_dims = (img_h, img_w)

            # Run YOLOv5 inference
            start_v5 = time.time()
            boxes5_norm, scores5, labels5 = predict_yolov5(yolov5_model, image_path, img_size)
            time_v5 = time.time() - start_v5
            # Denormalize v5 boxes for individual metric calculation
            boxes5 = (np.array(boxes5_norm) * np.array([img_w, img_h, img_w, img_h])).tolist() if boxes5_norm else []
            metrics_v5 = calculate_metrics((np.array(boxes5), np.array(scores5), np.array(labels5)), time_v5)
            all_metrics_v5.append(metrics_v5)

            # Run YOLOv8 inference
            start_v8 = time.time()
            boxes8_norm, scores8, labels8 = predict_yolov8(yolov8_model, image_path, img_size)
            time_v8 = time.time() - start_v8
            # Denormalize v8 boxes for individual metric calculation
            boxes8 = (np.array(boxes8_norm) * np.array([img_w, img_h, img_w, img_h])).tolist() if boxes8_norm else []
            metrics_v8 = calculate_metrics((np.array(boxes8), np.array(scores8), np.array(labels8)), time_v8)
            all_metrics_v8.append(metrics_v8)
            
            # Run WBF Ensemble
            start_ens = time.time()
            ens_boxes, ens_scores, ens_labels = run_wbf(
                [boxes5_norm, boxes8_norm],
                [scores5, scores8],
                [labels5, labels8],
                image_dims=image_dims,
                iou_thr=iou_thr_wbf,
                skip_box_thr=skip_box_thr_wbf
            )
            time_ens_wbf = time.time() - start_ens
            # Total ensemble time includes inference + WBF processing
            time_ens_total = time_v5 + time_v8 + time_ens_wbf 
            metrics_ensemble = calculate_metrics((ens_boxes, ens_scores, ens_labels), time_ens_total)
            all_metrics_ensemble.append(metrics_ensemble)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    # Aggregate metrics
    final_metrics = {}
    for name, metrics_list in zip(['yolov5', 'yolov8', 'ensemble_wbf'], 
                                  [all_metrics_v5, all_metrics_v8, all_metrics_ensemble]):
        if metrics_list:
             # Calculate averages, handling potential empty lists or metrics where num_detections is 0
            valid_metrics_for_conf = [m['average_confidence'] for m in metrics_list if m['num_detections'] > 0]
            valid_metrics_for_diam = [m['average_diameter_pixels'] for m in metrics_list if m['num_detections'] > 0]
            
            final_metrics[name] = {
                'average_inference_time_s': np.mean([m['inference_time'] for m in metrics_list]) if metrics_list else 0,
                'average_num_detections': np.mean([m['num_detections'] for m in metrics_list]) if metrics_list else 0,
                'overall_average_confidence': np.mean(valid_metrics_for_conf) if valid_metrics_for_conf else 0,
                'overall_average_diameter_pixels': np.mean(valid_metrics_for_diam) if valid_metrics_for_diam else 0
            }
        else:
             final_metrics[name] = {'inference_time': 0, 'num_detections': 0, 'average_confidence': 0, 'average_diameter_pixels': 0, 'error': 'No metrics collected'}


    # Save results
    os.makedirs('results/evaluation', exist_ok=True)
    output_path = 'results/evaluation/ensemble_metrics.json'
    try:
        with open(output_path, 'w') as f:
            yaml.dump(final_metrics, f, indent=4, default_flow_style=False)
        print(f"Evaluation complete. Results saved to {output_path}")
    except Exception as e:
        print(f"Error saving results to {output_path}: {e}")


if __name__ == "__main__":
    main() 