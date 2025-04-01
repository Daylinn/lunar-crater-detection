"""
My lunar crater detection evaluation script.
This script helps us understand how well our model is performing at detecting craters.
"""

import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from models.yolo_crater import YOLOv5
import yaml
from pathlib import Path
from yolov5.utils.general import check_file, increment_path
from yolov5.utils.metrics import ap_per_class
from yolov5.utils.plots import plot_pr_curve, plot_confusion_matrix
from yolov5.utils.torch_utils import select_device
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import create_dataloader

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

def evaluate_model(weights, data_yaml, imgsz=640, batch_size=32, conf_thres=0.001, iou_thres=0.6, device=''):
    """
    Evaluate our model's performance on the test set.
    This will give us metrics like precision, recall, and mAP.
    """
    # Set up our device
    device = select_device(device)
    
    # Load our model
    model = DetectMultiBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    
    # Load our dataset configuration
    data = check_file(data_yaml)
    with open(data, encoding='ascii', errors='ignore') as f:
        data = yaml.safe_load(f)
    
    # Create our test dataloader
    test_loader = create_dataloader(data['test'],
                                  imgsz=imgsz,
                                  batch_size=batch_size,
                                  stride=stride,
                                  pad=0.5,
                                  prefix=colorstr('val: '))[0]
    
    # Run evaluation
    model.eval()
    stats = []
    
    for batch_i, (im, targets, paths, shapes) in enumerate(test_loader):
        im = im.to(device).float() / 255.0
        targets = targets.to(device)
        
        # Forward pass
        pred = model(im)
        
        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        
        # Statistics
        for si, pred in enumerate(pred):
            labels = targets[targets[:, 0] == si, 1:]
            stats.append((pred, labels))
    
    # Compute metrics
    p, r, f1, mp, mr, map50, map = ap_per_class(*zip(*stats))
    
    # Print results
    print(f'\nResults:')
    print(f'Precision: {mp:.3f}')
    print(f'Recall: {mr:.3f}')
    print(f'F1-score: {f1:.3f}')
    print(f'mAP50: {map50:.3f}')
    print(f'mAP50-95: {map:.3f}')
    
    # Plot results
    plot_pr_curve(p, r, ap, Path('results/pr_curve.png'))
    plot_confusion_matrix(stats, save_dir=Path('results'))

def main():
    """Main function to run our evaluation."""
    # Set up our paths
    weights = 'runs/detect/train/weights/best.pt'
    data_yaml = 'data/lunar_craters.yaml'
    
    # Run evaluation
    evaluate_model(weights, data_yaml)

if __name__ == '__main__':
    main()