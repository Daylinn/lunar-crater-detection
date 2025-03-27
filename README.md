# Crater Detection with YOLOv5

A deep learning project for detecting craters in lunar images using YOLOv5 architecture.

## Project Structure

```
crater-detection-classification/
├── data/
│   └── raw/
│       └── images/          # Crater images
├── model/
│   ├── yolo_crater.py      # YOLOv5 model implementation
│   └── crater_detector.pth # Trained model weights
├── train.py                # Training script
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place your crater images in the `data/raw/images/` directory.

## Training

To train the model:

```bash
python train.py
```

The script will:

- Load and preprocess the images
- Train the YOLOv5 model
- Save the best model weights
- Display training metrics

## Model Architecture

The project uses a YOLOv5-based architecture with:

- CSP (Cross Stage Partial) backbone
- PANet neck
- Multi-scale detection heads
- Attention mechanisms for feature enhancement

## Data Format

Images should be in JPG format. The model expects:

- RGB images
- Single crater per image (centered)
- Standard YOLO format annotations

## Dependencies

- PyTorch
- TorchVision
- Albumentations
- OpenCV
- NumPy
- Pillow
- scikit-learn

## License

This project is licensed under the MIT License - see the LICENSE file for details.
