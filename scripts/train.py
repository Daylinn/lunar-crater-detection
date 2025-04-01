# scripts/train.py

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models.yolo_crater import YOLOv5  # Ensure this file is in the models/ folder

# Custom Dataset for crater detection (assumes YOLO format: class, x_center, y_center, width, height)
class CraterDataset(Dataset):
    def __init__(self, image_dir, label_dir, img_size=640, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load and preprocess image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image.astype(np.float32) / 255.0  # Normalize to [0,1]
        
        # Load label (YOLO format: class x_center y_center width height)
        base_name = os.path.splitext(self.image_files[idx])[0]
        label_path = os.path.join(self.label_dir, base_name + ".txt")
        with open(label_path, "r") as f:
            line = f.readline().strip().split()
            label = [float(x) for x in line]
        
        # Detection label: bounding box coordinates (x_center, y_center, w, h)
        detection_label = torch.tensor(label[1:], dtype=torch.float32)
        # Classification label: class id (e.g., 0 for crater)
        classification_label = torch.tensor(int(label[0]), dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
        else:
            # Convert to tensor with shape [C, H, W]
            image = torch.from_numpy(image).permute(2, 0, 1)
        
        return image, detection_label, classification_label

def train_model(model, dataloader, device, optimizer, det_criterion, cls_criterion):
    model.train()
    running_loss = 0.0
    for images, det_targets, cls_targets in dataloader:
        images = images.to(device)
        det_targets = det_targets.to(device)
        cls_targets = cls_targets.to(device)
        
        optimizer.zero_grad()
        # Forward pass: model outputs shape [B, num_bbox, grid_h, grid_w, (5+num_classes)]
        outputs = model(images)
        # For simplicity, assume each image contains one crater and use predictions from the first grid cell and bbox
        preds = outputs[:, 0, 0, 0, :]  # [B, 5+num_classes]
        bbox_preds = preds[:, :4]  # Bounding box predictions
        class_preds = preds[:, 5:]  # Class predictions
        
        loss_bbox = det_criterion(bbox_preds, det_targets)
        loss_cls = cls_criterion(class_preds, cls_targets)
        loss = loss_bbox + loss_cls
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_loss = running_loss / len(dataloader)
    return avg_loss

def main():
    # Training settings
    img_size = 640
    batch_size = 4
    epochs = 10
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare dataset and DataLoader for training data
    train_dataset = CraterDataset("data/train/images", "data/train/labels", img_size=img_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model (assumes one class: 'crater')
    model = YOLOv5(num_classes=1)
    model.to(device)
    
    # Define loss functions and optimizer
    det_criterion = nn.MSELoss()          # For bounding box regression
    cls_criterion = nn.CrossEntropyLoss()   # For classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        avg_loss = train_model(model, train_loader, device, optimizer, det_criterion, cls_criterion)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Save the trained model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/yolov5_crater.pth")
    print("Training complete. Model saved to checkpoints/yolov5_crater.pth")

if __name__ == "__main__":
    main()