# Lunar Crater Detection using YOLOv5

This project implements a deep learning model for detecting craters on the lunar surface using YOLOv5. The model is trained on a custom dataset of lunar images and can detect craters with high accuracy.

## Features

- Custom YOLOv5 implementation for lunar crater detection
- Efficient training pipeline with data augmentation
- Real-time inference capabilities
- Visualization tools for model predictions
- Comprehensive evaluation metrics

## Project Structure

```
lunar-crater-detection/
├── data/                    # Dataset directory
│   ├── train/              # Training images and labels
│   ├── valid/              # Validation images and labels
│   └── test/               # Test images and labels
├── scripts/                 # Python scripts
│   ├── train_yolo.py       # Training script
│   ├── predict.py          # Prediction script
│   ├── prepare_data.py     # Data preparation script
│   └── evaluate.py         # Evaluation script
├── models/                  # Model architecture definitions
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/lunar-crater-detection.git
cd lunar-crater-detection
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the model on your dataset:

```bash
cd scripts
python train_yolo.py
```

### Prediction

To make predictions on new images:

```bash
cd scripts
python predict.py
```

## Model Performance

The trained model achieves the following metrics:

- mAP50: 0.702 (70.2% mean average precision)
- Precision: 0.669 (66.9% of detections are correct)
- Recall: 0.663 (66.3% of craters are being detected)
- mAP50-95: 0.384 (38.4% mean average precision across different IoU thresholds)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv5 by Ultralytics
- Lunar Reconnaissance Orbiter Camera (LROC) for providing lunar imagery
- Contributors and maintainers of the open-source community
