import os
import torch
from pathlib import Path
from ultralytics import YOLO
import yaml

def create_yolo_config():
    """Create YOLO configuration file"""
    config = {
        'path': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')),  # dataset root dir
        'train': 'train/images',  # train images (relative to 'path')
        'val': 'valid/images',    # val images (relative to 'path')
        'test': 'test/images',    # test images (relative to 'path')

        'names': {0: 'crater'},   # class names
        'nc': 1,                  # number of classes

        # Training parameters - optimized for faster training
        'epochs': 50,             # reduced from 100 to 50
        'batch_size': 32,         # increased from 16 to 32
        'imgsz': 640,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # Model parameters - using smaller model for faster training
        'model': 'yolov5n.pt',    # changed from yolov5s.pt to yolov5n.pt (nano model)
        'pretrained': True,       # use pretrained model
        'optimizer': 'SGD',       # optimizer
        'lr0': 0.01,             # initial learning rate
        'momentum': 0.937,       # SGD momentum/Adam beta1
        'weight_decay': 0.0005,  # optimizer weight decay
        'warmup_epochs': 2,      # reduced from 3 to 2
        'warmup_momentum': 0.8,  # warmup initial momentum
        'warmup_bias_lr': 0.1,   # warmup initial bias lr
        'box': 7.5,              # box loss gain
        'cls': 0.5,              # cls loss gain
        'dfl': 1.5,              # dfl loss gain

        # Additional optimizations
        'cache': True,           # cache images in memory
        'workers': 4,            # reduced from 8 to 4 for CPU
        'amp': True,             # mixed precision training
    }
    
    # Save configuration
    config_path = os.path.join(os.path.dirname(__file__), 'yolo_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    return config

def train_model(config):
    """Train YOLOv5 model"""
    # Initialize model
    model = YOLO(config['model'])
    
    # Train the model
    results = model.train(
        data=os.path.join(os.path.dirname(__file__), 'yolo_config.yaml'),
        epochs=config['epochs'],
        batch=config['batch_size'],
        imgsz=config['imgsz'],
        device=config['device'],
        pretrained=config['pretrained'],
        optimizer=config['optimizer'],
        lr0=config['lr0'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay'],
        warmup_epochs=config['warmup_epochs'],
        warmup_momentum=config['warmup_momentum'],
        warmup_bias_lr=config['warmup_bias_lr'],
        box=config['box'],
        cls=config['cls'],
        dfl=config['dfl'],
        cache=config['cache'],
        workers=config['workers'],
        amp=config['amp']
    )
    
    return results

def validate_model(model_path, data_yaml):
    """Validate trained model"""
    model = YOLO(model_path)
    results = model.val(data=data_yaml)
    return results

def predict_image(model_path, image_path, conf_threshold=0.25):
    """Run inference on a single image"""
    model = YOLO(model_path)
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=True,
        save_txt=True,
        save_conf=True,
        show=True
    )
    return results

if __name__ == "__main__":
    # Create configuration
    config = create_yolo_config()
    
    # Train model
    print("Starting training...")
    results = train_model(config)
    
    # Validate model
    print("\nValidating model...")
    val_results = validate_model('runs/detect/train/weights/best.pt', os.path.join(os.path.dirname(__file__), 'yolo_config.yaml'))
    
    print("\nTraining and validation complete!")
    print(f"Best mAP50: {val_results.box.map50}")
    print(f"Best mAP50-95: {val_results.box.map}") 