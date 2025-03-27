# Lunar Crater Detection with YOLOv5

A deep learning project for detecting craters in lunar images using a custom YOLOv5 architecture with attention mechanisms.

## Project Structure

```
lunar-crater-detection/
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

## Team Tasks

### Team Member 1: Data Collection and Preprocessing

1. **Dataset Collection**

   - Gather lunar crater images from reliable sources
   - Ensure images are high quality and contain clear crater features
   - Target: 200-300 images minimum
   - Sources to consider:
     - NASA's Lunar Reconnaissance Orbiter Camera (LROC)
     - Apollo mission archives
     - Public lunar image datasets

2. **Data Organization**

   - Organize images in `data/raw/images/`
   - Create a data validation script
   - Document image sources and metadata

3. **Data Augmentation**
   - Test and tune the current augmentation pipeline
   - Add crater-specific augmentations if needed
   - Document augmentation parameters

### Team Member 2: Model Training and Optimization

1. **Training Pipeline**

   - Set up training environment
   - Run initial training with current dataset
   - Monitor and log training metrics
   - Implement model checkpointing

2. **Model Optimization**

   - Tune hyperparameters:
     - Learning rate
     - Batch size
     - Number of epochs
     - Early stopping patience
   - Experiment with different optimizer settings
   - Document all experiments and results

3. **Performance Analysis**
   - Implement evaluation metrics
   - Create visualization tools for:
     - Training curves
     - Detection results
     - Attention maps
   - Compare model performance across different scales

### Team Member 3: Evaluation and Deployment

1. **Model Evaluation**

   - Create a comprehensive test suite
   - Implement cross-validation
   - Test model on different image conditions
   - Document evaluation metrics

2. **Visualization and Analysis**

   - Create visualization tools for:
     - Detection results
     - False positives/negatives
     - Model confidence scores
   - Generate analysis reports

3. **Documentation and Presentation**
   - Create detailed technical documentation
   - Prepare presentation materials
   - Document model limitations and future improvements
   - Create user guide for model usage

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
- Custom attention mechanisms for feature enhancement
- Crater-specific augmentations

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
