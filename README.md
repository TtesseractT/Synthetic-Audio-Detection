# Multi-Head Binary Classification System for Synthetic Data Detection

## Overview
This repository contains the implementation of a **multi-head binary classification system** designed to detect AI-generated (synthetic) audio. The system utilizes an ensemble of neural network sub-models to distinguish between real and synthetic data. Key features include:

- **Multi-head neural network architecture** to improve classification accuracy.
- **Probabilistic modeling and novel output aggregation strategy**.
- **Comprehensive data processing pipeline** including augmentation and segmentation.
- **Modular structure**, allowing flexibility and scalability.

## Features
- **Multi-Head Model**: Uses multiple sub-models ("heads") for classification.
- **Ensemble Learning**: Outputs from sub-models are aggregated to enhance robustness.
- **Data Preprocessing**: Scripts for file renaming, conversion, augmentation, segmentation, and dataset management.
- **Efficient Inference**: Merged models enable fast predictions on new data.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- PyTorch
- FFmpeg (for audio processing)
- Librosa (for feature extraction)
- NumPy, Pandas, tqdm (for data processing)

### Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
### 1. Data Preprocessing
Run the following scripts in order to prepare your dataset:

#### File Renaming
```sh
python file_renamer.py --input-dir path/to/data
```
Ensures consistent and unique filenames.

#### Audio Conversion
```sh
python audio_converter.py --input-dir path/to/data --output-dir path/to/converted
```
Normalizes audio format and sampling rate.

#### Audio Augmentation
```sh
python audio_augment.py --input-dir path/to/converted --output-dir path/to/augmented
```
Generates augmented audio examples for improved training.

#### Audio Segmentation
```sh
python audio_segmenter.py --input-dir path/to/augmented --output-dir path/to/segmented
```
Splits audio files into fixed-length segments.

#### Dataset Management
```sh
python dataset_manager.py --input-dir path/to/segmented --output-dir path/to/dataset_split
```
Splits data into training and testing sets.

#### Overlap Checking
```sh
python file_manager.py --input-dir path/to/dataset_split
```
Ensures no data leakage between training and testing sets.

### 2. Model Training
Train the individual sub-models:
```sh
python submodel_trainer.py --train-dir path/to/dataset_split/train --epochs 50
```

### 3. Model Merging
Merge trained sub-models into a unified multi-head model:
```sh
python model_merger.py --model-dir path/to/trained_models --output path/to/merged_model.pth
```

### 4. Inference
Run the trained model on new audio data:
```sh
python inference_runner.py --model path/to/merged_model.pth --input path/to/audio
```

## Model Architecture
The classification model consists of multiple convolutional neural network (CNN) sub-models. Each sub-model:
- Extracts features from audio spectrograms using a **ResNet-18** backbone.
- Outputs logits for "Real" and "Synthetic" classes.
- The ensemble averages the **Real** logits and retains individual **Synthetic** logits.
- Final classification is based on a thresholded decision rule.

## Performance
- **Accuracy**: Expected to reach **95-99%** on well-prepared datasets.
- **Scalability**: Easily extends by adding new sub-models for additional synthetic classes.
- **Efficiency**: Supports GPU acceleration for fast inference.
- **Robustness**: Designed to reduce false negatives in synthetic audio detection.

## Future Work
- Implement weight-sharing between sub-models.
- Extend detection to additional synthetic data modalities (e.g., video, text).
- Optimize inference pipeline for real-time applications.

## Citation
If you use this work in your research, please cite:
```
@article{Hibbs2025,
  author = {Sabian Hibbs},
  title = {Multi-Head Binary Classification System for Synthetic Data Detection},
  year = {2025},
  journal = {arXiv}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
