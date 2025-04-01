"""
A simple script to detect craters in lunar images and show the results.
"""

import cv2
from ultralytics import YOLO
import os

def show_detection(image_path, model_path, conf_threshold=0.25):
    # Load the model
    model = YOLO(model_path)
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    # Run prediction
    results = model.predict(img, conf=conf_threshold, save=False)
    
    # Get the first result
    result = results[0]
    
    # Draw boxes and labels
    for box in result.boxes:
        # Get box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Get confidence
        conf = float(box.conf[0])
        
        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label
        label = f"Crater {conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Show the image
    window_name = "Lunar Crater Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)
    
    # Wait for a key press
    print("Press any key to close the window...")
    cv2.waitKey(0)
    
    # Close the window
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Path to the model weights
    model_path = "runs/detect/train3/weights/best.pt"
    
    # Path to the test image
    image_path = "../data/test/images/mars_crater--97-_jpg.rf.63347c2ec963cf5c4ab641e1ba872df1.jpg"
    
    # Show detection
    show_detection(image_path, model_path) 