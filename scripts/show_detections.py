"""
A simple script to detect craters in lunar images and show the results.
"""

import cv2
from ultralytics import YOLO
import os
import random

def get_random_test_image():
    """Get a random image from the test dataset."""
    test_dir = "../data/test/images"
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found at {test_dir}")
        return None
    
    # Get list of all image files
    image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print("Error: No images found in test directory")
        return None
    
    # Select random image
    random_image = random.choice(image_files)
    return os.path.join(test_dir, random_image)

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
    
    # Get a random test image
    image_path = get_random_test_image()
    if image_path:
        print(f"Processing image: {os.path.basename(image_path)}")
        show_detection(image_path, model_path)
    else:
        print("Failed to find a test image to process.") 