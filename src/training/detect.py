from ultralytics import YOLO
import cv2
import os
import argparse
import json
from datetime import datetime
import time
from pathlib import Path
import numpy as np

def preprocess_image(image_path, target_size=(640, 640)):
    """Preprocess image for YOLOv8 model."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, target_size)
    
    return img

def detect_craters(model, image_path, conf_threshold=0.25, save_txt=False, save_conf=False):
    """Detect craters in an image using YOLOv8 model."""
    # Preprocess image
    img = preprocess_image(image_path)
    
    # Run detection
    start_time = time.time()
    results = model(img, conf=conf_threshold)
    detection_time = time.time() - start_time
    
    # Get detections
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(conf)
            })
    
    # Save detections to txt file if requested
    if save_txt:
        txt_path = os.path.splitext(image_path)[0] + '.txt'
        with open(txt_path, 'w') as f:
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                conf = det['confidence']
                # Convert to YOLO format (normalized center x, center y, width, height)
                img_h, img_w = img.shape[:2]
                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h
                if save_conf:
                    f.write(f"0 {x_center} {y_center} {width} {height} {conf}\n")
                else:
                    f.write(f"0 {x_center} {y_center} {width} {height}\n")
    
    return detections, detection_time

def main():
    parser = argparse.ArgumentParser(description='Detect craters in images using YOLOv8')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--source', type=str, required=True, help='Path to source images')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--save-txt', action='store_true', help='Save detections to txt files')
    parser.add_argument('--save-conf', action='store_true', help='Save confidence scores in txt files')
    args = parser.parse_args()
    
    # Load model
    model = YOLO(args.weights)
    
    # Process source
    if os.path.isdir(args.source):
        image_paths = [os.path.join(args.source, f) for f in os.listdir(args.source) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        image_paths = [args.source]
    
    total_detections = 0
    total_time = 0
    
    for image_path in image_paths:
        detections, detection_time = detect_craters(
            model, 
            image_path, 
            conf_threshold=args.conf,
            save_txt=args.save_txt,
            save_conf=args.save_conf
        )
        total_detections += len(detections)
        total_time += detection_time
        
        print(f"Processed {image_path}:")
        print(f"  Detections: {len(detections)}")
        print(f"  Time: {detection_time:.3f}s")
        if detections:
            avg_conf = sum(d['confidence'] for d in detections) / len(detections)
            print(f"  Average confidence: {avg_conf:.3f}")
    
    print("\nSummary:")
    print(f"Total images processed: {len(image_paths)}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {total_detections/len(image_paths):.2f}")
    print(f"Total processing time: {total_time:.3f}s")
    print(f"Average time per image: {total_time/len(image_paths):.3f}s")

if __name__ == "__main__":
    main() 