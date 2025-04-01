# scripts/evaluate.py

import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from models.yolo_crater import YOLOv5

# Custom Dataset for evaluation (only loads images)
class EvalDataset(Dataset):
    def __init__(self, image_dir, img_size=640, transform=None):
        self.image_dir = image_dir
        self.img_size = img_size
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image, self.image_files[idx]

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the trained model checkpoint
    model = YOLOv5(num_classes=1)
    checkpoint_path = "checkpoints/yolov5_crater.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Prepare evaluation dataset and DataLoader
    eval_dataset = EvalDataset("data/test/images", img_size=640)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
    
    for image, img_name in eval_loader:
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image)
        # For simplicity, use the first predicted bounding box from the first grid cell
        preds = outputs[:, 0, 0, 0, :]  # [B, 5+num_classes]
        bbox_preds = preds[:, :4].cpu().numpy()[0]  # [x, y, w, h]
        
        # Convert image to numpy for plotting
        img_np = image.cpu().squeeze().permute(1, 2, 0).numpy()
        
        # Assume coordinates are normalized; convert to pixel values
        h, w, _ = img_np.shape
        x_center, y_center, box_w, box_h = bbox_preds
        x_center *= w
        y_center *= h
        box_w *= w
        box_h *= h
        
        # Calculate top-left corner coordinates
        x1 = int(x_center - box_w / 2)
        y1 = int(y_center - box_h / 2)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(img_np)
        rect = plt.Rectangle((x1, y1), int(box_w), int(box_h), edgecolor='red', facecolor='none', linewidth=2)
        plt.gca().add_patch(rect)
        plt.title(f"Prediction: {img_name[0]}")
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    evaluate()