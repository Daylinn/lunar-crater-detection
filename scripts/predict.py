from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def predict():
    # Load the trained model
    model_path = "runs/detect/train3/weights/best.pt"
    if not Path(model_path).exists():
        model_path = "runs/detect/train3/weights/last.pt"
    
    # Load the model
    model = YOLO(model_path)
    
    # Get test images
    test_dir = Path("../data/test/images")
    if not test_dir.exists():
        print("Test directory not found. Please ensure data/test/images exists.")
        return
    
    # Process each image
    for img_path in test_dir.glob("*.jpg"):
        # Read and process image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Make prediction
        results = model(img, conf=0.25)[0]
        
        # Plot results
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        
        # Draw predictions
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               edgecolor='red', facecolor='none', 
                               linewidth=2, label=f'Crater ({conf:.2f})')
            plt.gca().add_patch(rect)
        
        plt.title(f"Predictions for {img_path.name}")
        plt.axis('off')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    predict() 