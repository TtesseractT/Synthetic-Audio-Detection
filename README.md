# Audio Classification Pipeline: Real vs Synthetic Detection

This repository contains an end-to-end solution for audio classification focused on distinguishing between real and synthetic audio signals. The project implements a comprehensive pipeline for audio processing, data augmentation, model training, and inference using state-of-the-art deep learning techniques.

## Overview

The project is organized into several key components:

- **Audio Augmentation:**  
  Enhance your dataset with multiple augmentation techniques—time stretching, pitch shifting, dynamic range compression, noise addition, tremolo, phaser, time shifting, and more—to improve model generalization.  
  *(See [audio_augmneter_batch.py](&#8203;:contentReference[oaicite:0]{index=0}))*

- **Audio Conversion & Segmentation:**  
  Standardize audio inputs by converting various formats (MP3, WAV, FLAC, etc.) to a uniform WAV format with a consistent sample rate and mono channel. Additionally, long audio files are segmented into fixed-duration chunks (e.g., 4-second segments) using ffmpeg.  
  *(See [audio_convert_wave.py](&#8203;:contentReference[oaicite:1]{index=1}) and [audio_convert_segmenter.py](&#8203;:contentReference[oaicite:2]{index=2}))*

- **File Renaming:**  
  Ensure unique file identification by renaming audio files based on the first 6 characters of their SHA256 hash.  
  *(See [file_rename_hash.py](&#8203;:contentReference[oaicite:3]{index=3}))*

- **Model Training:**  
  Train individual binary classifiers (sub-models) on spectrogram data derived from audio files. The training script incorporates data augmentation, logging, checkpointing, and evaluation routines.  
  *(See [train_submodel.py](&#8203;:contentReference[oaicite:4]{index=4}))*

- **Model Merging:**  
  Merge several sub-models into a unified multi-head classifier. This merged model leverages the outputs of individual sub-models—each providing a separate synthetic prediction—and averages their real predictions for robust inference.  
  *(See [merge_model_classifier.py](&#8203;:contentReference[oaicite:5]{index=5}))*

- **Inference Pipeline:**  
  Process new audio files using an overlapping window approach, convert them into spectrograms, and classify each segment with the merged model. The pipeline supports probability smoothing and thresholding for reliable decision-making.  
  *(See [inference_classifier.py](&#8203;:contentReference[oaicite:6]{index=6}))*

An example output of the inference process is provided in the `results.json` file.  
*(See [results.json](&#8203;:contentReference[oaicite:7]{index=7}))*

## Repository Structure

.
├── audio_augmneter_batch.py      # Audio data augmentation script
├── audio_convert_segmenter.py    # Audio segmentation using ffmpeg
├── audio_convert_wave.py         # Audio conversion to standardized WAV format
├── file_rename_hash.py           # File renaming based on SHA256 hash
├── inference_classifier.py       # Inference script using the merged multi-head classifier
├── merge_model_classifier.py     # Merges sub-models into a multi-head classifier
├── train_submodel.py             # Training script for binary classifiers on spectrogram data
├── results.json                  # Sample output from the inference process
└── README.md                     # This README file

## Getting Started

### Prerequisites

- **Python 3.x**
- **PyTorch** and **torchaudio**
- **librosa**
- **timm**
- **ffmpeg** (command-line tool)
- Other required Python packages: numpy, pandas, tqdm, argparse, etc.

### Installation

1. **Clone the Repository:**

   git clone https://github.com/yourusername/your-repo.git
   cd your-repo

2. **Create and Activate a Virtual Environment (Optional):**

   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install Dependencies:**

   pip install -r requirements.txt

### Usage

- **Audio Augmentation:**

  python audio_augmneter_batch.py -i <input_audio_or_folder> -o <output_folder>

- **Audio Conversion & Segmentation:**

  python audio_convert_wave.py -i <input_folder> -o <output_folder>
  python audio_convert_segmenter.py -i <input_file_or_folder> -o <output_folder>

- **File Renaming:**

  python file_rename_hash.py -i <directory> [-r]

- **Training a Sub-model:**

  python train_submodel.py --data-dir <dataset_directory> [other options]

- **Merging Sub-models:**

  Prepare a CSV file with the columns model_filename, synthetic_class, and real_class. Then run:

  python merge_model_classifier.py --submodels-folder <folder> --csv-file <csv_file> --output-path <merged_model.pth>

- **Inference:**

  python inference_classifier.py --merged-model <merged_model.pth> --audio <audio_file.wav> [--threshold 0.5 --smooth]

## Contributing

Contributions are welcome! Please submit issues or pull requests with improvements, bug fixes, or suggestions.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- This project leverages powerful libraries such as PyTorch, torchaudio, librosa, and timm.
- Special thanks to the open-source community for providing the tools and resources that made this project possible.

## Citations

- audio_augmneter_batch.py: :contentReference[oaicite:8]{index=8}
- audio_convert_segmenter.py: :contentReference[oaicite:9]{index=9}
- audio_convert_wave.py: :contentReference[oaicite:10]{index=10}
- file_rename_hash.py: :contentReference[oaicite:11]{index=11}
- inference_classifier.py: :contentReference[oaicite:12]{index=12}
- merge_model_classifier.py: :contentReference[oaicite:13]{index=13}
- results.json: :contentReference[oaicite:14]{index=14}
- train_submodel.py: :contentReference[oaicite:15]{index=15}
