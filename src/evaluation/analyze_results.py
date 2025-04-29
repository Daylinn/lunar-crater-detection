"""
Analyze existing YOLOv5 and YOLOv8 detection results.
"""

import os
import json
from glob import glob
import cv2
import numpy as np

def count_detections(image_path):
    """Count detections in an image by looking for red bounding boxes."""
    img = cv2.imread(image_path)
    if img is None:
        return 0
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define red color range
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    
    # Create mask and count contours
    mask = cv2.inRange(hsv, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return len(contours)

def analyze_results():
    """Analyze detection results from both models."""
    # Paths to results
    yolov5_path = "results/yolov5"
    yolov8_path = "results/yolov8/detection_results_20250410_170653/results"
    
    # Get image lists
    yolov5_images = glob(os.path.join(yolov5_path, "*.jpg"))
    yolov8_images = glob(os.path.join(yolov8_path, "*.jpg"))
    
    # Analyze results
    results = {
        "yolov5": {
            "total_images": len(yolov5_images),
            "total_detections": 0,
            "detections_per_image": []
        },
        "yolov8": {
            "total_images": len(yolov8_images),
            "total_detections": 0,
            "detections_per_image": []
        }
    }
    
    print("Analyzing YOLOv5 results...")
    for img_path in yolov5_images:
        detections = count_detections(img_path)
        results["yolov5"]["total_detections"] += detections
        results["yolov5"]["detections_per_image"].append(detections)
    
    print("Analyzing YOLOv8 results...")
    for img_path in yolov8_images:
        detections = count_detections(img_path)
        results["yolov8"]["total_detections"] += detections
        results["yolov8"]["detections_per_image"].append(detections)
    
    # Calculate averages
    for model in ["yolov5", "yolov8"]:
        if results[model]["total_images"] > 0:
            results[model]["avg_detections_per_image"] = (
                results[model]["total_detections"] / results[model]["total_images"]
            )
    
    # Save results
    os.makedirs("results/analysis", exist_ok=True)
    with open("results/analysis/detection_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\nAnalysis Summary:")
    print("-" * 50)
    for model in ["yolov5", "yolov8"]:
        print(f"\n{model.upper()} Results:")
        print(f"Total images processed: {results[model]['total_images']}")
        print(f"Total detections: {results[model]['total_detections']}")
        if results[model]["total_images"] > 0:
            print(f"Average detections per image: {results[model]['avg_detections_per_image']:.2f}")
    
    return results

if __name__ == "__main__":
    analyze_results() 