# Lunar Crater Detection

This project was developed as part of our Master's in AI program, focusing on computer vision and deep learning applications in space science. We developed a system to automatically detect and classify craters on the lunar surface using YOLOv5, a state-of-the-art object detection model.

## Project Overview

Our system leverages deep learning to identify and localize craters in lunar imagery. We trained a YOLOv5 model on a custom dataset of lunar images, achieving robust crater detection across various lighting conditions and crater sizes. This work demonstrates the practical application of computer vision techniques to space science research.

## Team Members

- Daylin Hart - Lead Developer
- David Williams - Contributor
- Lamont Carter - Contributor
- Sasi Pavan Khadyoth Gunturu - Contributor

## Technical Implementation

The project is structured as follows:

- `data/`: Contains our curated dataset of lunar images and annotations
- `models/`: Stores the trained model weights
- `scripts/`: Implementation of training and inference pipelines
- `notebooks/`: Analysis and experimentation notebooks

## Getting Started

1. Clone the repository:

```bash
git clone https://github.com/Daylinn/lunar-crater-detection.git
cd lunar-crater-detection
```

2. Set up the environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Run inference:

```bash
cd scripts
python show_detections.py
```

This will demonstrate the model's performance on a random test image.

## Methodology

1. **Data Collection and Preparation**: We curated a dataset of lunar images and manually annotated crater locations
2. **Model Development**: Implemented YOLOv5 with custom configurations for lunar crater detection
3. **Training and Validation**: Trained the model using our dataset and validated performance across different image conditions
4. **Testing and Evaluation**: Conducted comprehensive testing to ensure robust performance

## Results

Our model demonstrates strong performance in crater detection:

- Successfully identifies craters across various sizes and lighting conditions
- Real-time inference capabilities
- Robust performance on previously unseen images

## Future Work

Potential areas for future development:

- Integration with additional lunar datasets
- Enhancement of detection accuracy for smaller craters
- Application to other celestial bodies
- Development of a web interface for easier interaction

## Contributing

We welcome contributions from the research community! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) file for detailed guidelines on how to contribute to the project. This includes:

- Setting up your development environment
- Code style guidelines
- Pull request process
- Areas for contribution
- Communication guidelines

## License

This project is open source under the MIT License, encouraging further research and development in the field.

## Acknowledgments

We would like to thank:

- The YOLOv5 team for their excellent framework
- The lunar research community for providing the imagery
- Our program advisors and peers for their valuable feedback

For questions or collaboration inquiries, please reach out to the team.
