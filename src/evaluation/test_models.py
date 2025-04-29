import os
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
import time
from tqdm import tqdm

class ModelTester:
    def __init__(self, test_dir='data/test/images'):
        self.test_dir = Path(test_dir)
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        
        # Load our custom-trained models
        self.model_v5 = YOLO('runs/detect/train/weights/best.pt')  # YOLOv5 model
        self.model_v8 = YOLO('runs/detect/train/weights/best.pt')  # YOLOv8 model
        
        # Performance metrics
        self.metrics = {
            'v5': {'time': [], 'detections': [], 'confidence': []},
            'v8': {'time': [], 'detections': [], 'confidence': []}
        }
    
    def preprocess_image(self, image):
        """Preprocess image for both models"""
        # Convert to RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to 640x640
        image = cv2.resize(image, (640, 640))
        
        return image
    
    def run_detection(self, model, image, model_name):
        """Run detection and measure performance"""
        start_time = time.time()
        
        # Run detection
        results = model(image, conf=0.25)
        
        # Calculate metrics
        detection_time = time.time() - start_time
        num_detections = len(results[0].boxes)
        avg_confidence = np.mean([box.conf.cpu().numpy() for box in results[0].boxes]) if num_detections > 0 else 0
        
        # Store metrics
        self.metrics[model_name]['time'].append(detection_time)
        self.metrics[model_name]['detections'].append(num_detections)
        self.metrics[model_name]['confidence'].append(avg_confidence)
        
        return results
    
    def visualize_results(self, image, results_v5, results_v8, save_path):
        """Visualize and compare results from both models"""
        # Create output image
        output = np.zeros((640, 1280, 3), dtype=np.uint8)
        
        # Draw YOLOv5 results
        v5_image = image.copy()
        for box in results_v5[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cv2.rectangle(v5_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(v5_image, f'v5: {conf:.2f}', (int(x1), int(y1)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw YOLOv8 results
        v8_image = image.copy()
        for box in results_v8[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cv2.rectangle(v8_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(v8_image, f'v8: {conf:.2f}', (int(x1), int(y1)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Combine images
        output[:640, :640] = v5_image
        output[:640, 640:] = v8_image
        
        # Add labels
        cv2.putText(output, 'YOLOv5', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(output, 'YOLOv8', (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Save result
        cv2.imwrite(str(save_path), output)
    
    def print_metrics(self):
        """Print performance metrics"""
        print("\nPerformance Comparison:")
        print("=" * 50)
        
        for model in ['v5', 'v8']:
            avg_time = np.mean(self.metrics[model]['time'])
            avg_detections = np.mean(self.metrics[model]['detections'])
            avg_confidence = np.mean(self.metrics[model]['confidence'])
            
            print(f"\nYOLO{model.upper()}:")
            print(f"Average detection time: {avg_time:.4f} seconds")
            print(f"Average detections per image: {avg_detections:.2f}")
            print(f"Average confidence: {avg_confidence:.4f}")
        
        print("\n" + "=" * 50)
    
    def test_models(self):
        """Run comparison test on all test images"""
        # Create output directory
        output_dir = Path('test_results')
        output_dir.mkdir(exist_ok=True)
        
        # Get test images
        test_images = list(self.test_dir.glob('*.jpg'))
        
        print(f"Testing on {len(test_images)} images...")
        
        for img_path in tqdm(test_images):
            # Load and preprocess image
            image = cv2.imread(str(img_path))
            image = self.preprocess_image(image)
            
            # Run detection with both models
            results_v5 = self.run_detection(self.model_v5, image, 'v5')
            results_v8 = self.run_detection(self.model_v8, image, 'v8')
            
            # Visualize and save results
            save_path = output_dir / f"comparison_{img_path.name}"
            self.visualize_results(image, results_v5, results_v8, save_path)
        
        # Print metrics
        self.print_metrics()

def main():
    tester = ModelTester()
    tester.test_models()

if __name__ == "__main__":
    main() 