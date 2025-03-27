import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from data.dataset import CraterDataset
from model.yolo_crater import CraterYOLO
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score

def compute_loss(predictions, targets):
    """Compute YOLO loss."""
    loss = 0
    for pred, target in zip(predictions, targets):
        # Split predictions into boxes, objectness, and class scores
        pred_boxes = pred[..., :4]
        pred_obj = pred[..., 4]
        pred_cls = pred[..., 5:]
        
        # Compute box loss (MSE)
        box_loss = nn.MSELoss()(pred_boxes, target[..., :4])
        
        # Compute objectness loss (BCE)
        obj_loss = nn.BCELoss()(pred_obj, target[..., 4])
        
        # Compute class loss (BCE)
        cls_loss = nn.BCELoss()(pred_cls, target[..., 5:])
        
        # Combine losses
        loss += box_loss + obj_loss + cls_loss
    
    return loss

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    for images, targets in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        predictions = model(images)
        loss = compute_loss(predictions, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(predictions[0].detach().cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    precision, recall, _ = precision_recall_curve(all_targets[..., 4].flatten(), all_preds[..., 4].flatten())
    ap = average_precision_score(all_targets[..., 4].flatten(), all_preds[..., 4].flatten())
    
    return total_loss / len(dataloader), ap

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            targets = targets.to(device)
            
            predictions = model(images)
            loss = compute_loss(predictions, targets)
            
            total_loss += loss.item()
            all_preds.extend(predictions[0].cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    precision, recall, _ = precision_recall_curve(all_targets[..., 4].flatten(), all_preds[..., 4].flatten())
    ap = average_precision_score(all_targets[..., 4].flatten(), all_preds[..., 4].flatten())
    
    return total_loss / len(dataloader), ap

def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize model
    model = CraterYOLO(num_classes=1)  # 1 class for craters
    model = model.to(device)
    
    # Create datasets
    train_dataset = CraterDataset(
        images_dir=Path("data/raw/images"),
        transform=CraterDataset.get_train_transform()
    )
    
    val_dataset = CraterDataset(
        images_dir=Path("data/raw/images"),
        transform=CraterDataset.get_val_transform()
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    # Training settings
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    num_epochs = 50
    best_val_ap = 0
    patience = 10
    patience_counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_ap = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_loss, val_ap = validate(model, val_loader, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Train AP: {train_ap:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val AP: {val_ap:.4f}")
        
        # Save best model
        if val_ap > best_val_ap:
            best_val_ap = val_ap
            torch.save(model.state_dict(), "model/crater_detector.pth")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    print("\nTraining completed!")
    print(f"Best validation AP: {best_val_ap:.4f}")

if __name__ == "__main__":
    main() 