"""
Basic analysis of detection results - file counts and sizes only.
"""

import os
from glob import glob
import json
from datetime import datetime

def analyze_basic_results():
    """Analyze basic metrics from detection results."""
    # Paths to results
    yolov5_path = "results/yolov5"
    yolov8_path = "results/yolov8/detection_results_20250410_170653/results"
    
    # Get file lists
    yolov5_files = glob(os.path.join(yolov5_path, "*.jpg"))
    yolov8_files = glob(os.path.join(yolov8_path, "*.jpg"))
    
    # Analyze results
    results = {
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "yolov5": {
            "total_files": len(yolov5_files),
            "file_sizes": [],
            "total_size_mb": 0
        },
        "yolov8": {
            "total_files": len(yolov8_files),
            "file_sizes": [],
            "total_size_mb": 0
        }
    }
    
    # Analyze YOLOv5 files
    print("Analyzing YOLOv5 results...")
    for file_path in yolov5_files:
        size_mb = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
        results["yolov5"]["file_sizes"].append({
            "file": os.path.basename(file_path),
            "size_mb": size_mb
        })
        results["yolov5"]["total_size_mb"] += size_mb
    
    # Analyze YOLOv8 files
    print("Analyzing YOLOv8 results...")
    for file_path in yolov8_files:
        size_mb = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
        results["yolov8"]["file_sizes"].append({
            "file": os.path.basename(file_path),
            "size_mb": size_mb
        })
        results["yolov8"]["total_size_mb"] += size_mb
    
    # Calculate averages
    for model in ["yolov5", "yolov8"]:
        if results[model]["total_files"] > 0:
            results[model]["avg_file_size_mb"] = (
                results[model]["total_size_mb"] / results[model]["total_files"]
            )
    
    # Save results
    os.makedirs("results/analysis", exist_ok=True)
    with open("results/analysis/basic_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\nBasic Analysis Summary:")
    print("-" * 50)
    for model in ["yolov5", "yolov8"]:
        print(f"\n{model.upper()}:")
        print(f"Total files: {results[model]['total_files']}")
        print(f"Total size: {results[model]['total_size_mb']:.2f} MB")
        if results[model]["total_files"] > 0:
            print(f"Average file size: {results[model]['avg_file_size_mb']:.2f} MB")
    
    return results

if __name__ == "__main__":
    analyze_basic_results() 