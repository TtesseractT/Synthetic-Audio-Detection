<div align="center">
    <h1>Synthetic Audio Detection System</h1>
    <p>A multi-head binary classification approach for detecting AI-generated audio</p>
    <a href="./Uhmbrella_Technical_Whitepaper.pdf">
        <img src="https://img.shields.io/badge/Whitepaper-PDF-red.svg" alt="Technical Whitepaper">
    </a>
    <a href="https://github.com/TtesseractT/Synthetic-Audio-Detection/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
    </a>
    <a href="https://github.com/TtesseractT/Synthetic-Audio-Detection">
        <img src="https://img.shields.io/github/stars/TtesseractT/Synthetic-Audio-Detection?style=social" alt="GitHub stars">
    </a>
</div>

<p align="center">
    <a href="./Uhmbrella_Technical_Whitepaper.pdf">
        <img src="https://img.shields.io/badge/Download-Technical_Whitepaper-blue?style=for-the-badge&logo=adobe-acrobat-reader" alt="Download Technical Whitepaper">
    </a>
</p>

---

A multi-head binary classification system for detecting and classifying synthetic audio, featuring a shared feature-learning backbone and an ensemble of sub-models. This README aims to provide a thorough, overview of the entire pipeline, complete with illustrative Python code snippets, usage examples, and references to the theoretical background described in the associated whitepaper.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Accuracy Metrics](#accuracy-metrics)
4. [Scripts and Workflow](#scripts-and-workflow)
   1. [File Renamer](#1-file-renamer-filerenamerpy)
   2. [Audio Converter](#2-audio-converter-audioconverterpy)
   3. [Audio Augmenter](#3-audio-augmenter-audio_augmentpy)
   4. [Audio Segmenter](#4-audio-segmenter-audio_segmenterpy)
   5. [Dataset Manager](#5-dataset-manager-dataset_managerpy)
   6. [File Manager (Overlap Checker)](#6-file-manager-overlap-checker-file_managerpy)
   7. [Submodel Trainer](#7-submodel-trainer-submodel_trainerpy)
   8. [Model Merger](#8-model-merger-model_mergerpy)
   9. [Inference Runner](#9-inference-runner-inference_runnerpy)
5. [Sample Inference Output](#sample-inference-output)
6. [Code Requirements](#code-requirements)
7. [License](#license)
8. [Citation](#citation)

---

## Introduction

This system aims to robustly distinguish **Real** audio from multiple types of **Synthetic** audio through an **ensemble** of binary classifiers. Each sub-model identifies whether an audio clip is real or one particular synthetic type, and all sub-model outputs are aggregated in a unique way: the real logits are averaged to ensure consensus for a *Real* classification, while any strong synthetic indicator from a sub-model can flag the clip as *Synthetic*.

The pipeline is subdivided into multiple scripts that cover every step of data preparation, model training, and inference, allowing a flexible and modular approach to synthetic audio detection.

## Architecture Overview

At a high level, the pipeline transforms raw audio data into 4-second, single-channel segments at 32 kHz. Each segment is turned into a mel-spectrogram and passed through sub-models. Each sub-model is a ResNet-based CNN producing two logits: `[Real, Synthetic]`. During **merging**, the real logits from all heads are averaged, forming an **ensemble** that allows for robust detection:

```
┌──────────────────────────── Shared CNN Backbone ─────────────────────────────┐
│                                                                              │
│                (Spectrogram) -> [Conv Layers + Blocks] -> Features           │
│                                                                              │
├────────────────────────────────────┬─────────────────────────────────────────┤
│ Sub-model #1 (Binary: Real vs S1)  │   Sub-model #2 (Binary: Real vs S2)     │
│ [2-Logit Output: z1_real, z1_syn]  │   [2-Logit Output: z2_real, z2_syn]     │
│               ...                  │                 ...                     │
├────────────────────────────────────┴─────────────────────────────────────────┤
│ Final Ensemble: [syn1, syn2, ... synN, mean_of_all_real_logits]              │
└──────────────────────────────────────────────────────────────────────────────┘

```
---

## Accuracy Metrics

Below is a summary of the classification metrics obtained during internal testing of 3.8 Million 4 second audio files with 6 indevidual classes:

> **Saved best model with accuracy:** 98.53%  
> **Epoch:** 3

| Class             | Precision | Recall | F1-Score |
|-------------------|-----------|--------|----------|
| **Overall Accuracy**  |         |        | **0.98** |
| **Macro Average**     | 0.98    | 0.97   | 0.98     |
| **Weighted Average**  | 0.98    | 0.98   | 0.98     |

---

## Scripts and Workflow

Below is a recommended order of script usage along with illustrative code snippets from the source files. Each script has its own command-line interface.

---

### 1. File Renamer (`file_renamer.py`)

**Purpose**: Renames audio files to a unique, hash-based filename.

- **Key Function**: `hash_file(file_path)`

```python
import hashlib

def hash_file(file_path):
    """Generate a hash for a file and return the first 16 characters."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as afile:
        buf = afile.read()
        hasher.update(buf)
    return hasher.hexdigest()[:16]
```

- **Usage**:
```bash
python file_renamer.py -i /path/to/audio -r
```

When run, this script recursively scans `/path/to/audio` and renames each file (e.g., `myFile.wav` -> `9af81b232c9aa123.wav`) so that all future processing steps remain consistent.

---

### 2. Audio Converter (`audio_converter.py`)

**Purpose**: Converts raw audio from various formats/bitrates to a standard WAV (32kHz, mono, 16-bit).

- **Notable Snippet**:
```python
command = [
    'ffmpeg',
    '-i', input_file,
    '-ar', '32000',  # Sample rate: 32000 Hz
    '-ac', '1',      # Mono channel
    '-sample_fmt', 's16',
    '-f', 'wav', output_file
]
subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
```

- **Usage**:
```bash
python audio_convert.py -i ./raw_audio -o ./converted_audio
```

The converter ensures uniform audio parameters, essential for consistent downstream processing and spectrogram generation.

---

### 3. Audio Augmenter (`audio_augment.py`)

**Purpose**: Applies various transformations (noise addition, pitch/time shifting, compression, etc.) to expand dataset diversity.

- **Key Techniques**:
  - **Time Stretching** (speed up/slow down)
  - **Pitch Shifting** (±2 semitones)
  - **Add White Noise**
  - **Random Filtering**

- **Example: Augmenting with White Noise** (simplified code excerpt):
```python
def augment_add_white_noise(y, min_vol=0.001, max_vol=0.05):
    rms = np.sqrt(np.mean(y ** 2))
    noise_amp = np.random.uniform(min_vol, max_vol) * rms
    noise = noise_amp * np.random.normal(size=y.shape[0])
    return y + noise, noise_amp
```

- **Usage**:
```bash
python audio_augmneter.py -i ./converted_audio/sample.wav -o ./augmented
```

This script can generate multiple augmented variants (e.g., 10) per original audio, labeled with descriptive suffixes (e.g., `..._add_white_noise_0.004.wav`).

---

### 4. Audio Segmenter (`audio_segmenter.py`)

**Purpose**: Splits audio into uniform 4-second clips (mono, 32kHz).

- **Core FFmpeg Segmenting**:
```python
ffmpeg_cmd = [
    'ffmpeg', '-loglevel', 'error', '-i', input_filepath,
    '-ac', '1',
    '-ar', '32000',
    '-filter:a', 'pan=mono|c0=0.5*c0+0.5*c1',
    '-f', 'segment',
    '-segment_time', str(segment_duration),
    '-reset_timestamps', '1',
    segment_pattern
]
subprocess.run(ffmpeg_cmd, check=True)
```

- **Usage**:
```bash
python audio_segmenter.py -i ./augmented -o ./segmented
```

Generates 4-second chunks like `abcdef1234567890_Segment_000.wav`, `_Segment_001.wav`, etc.

---

### 5. Dataset Manager (`dataset_manager.py`)

**Purpose**: Creates structured `train/` and `test/` splits by random sampling each class folder.

- **Random File Splitting** example:
```python
files = [f for f in os.listdir(source_folder) if f.lower().endswith(".wav")]
num_train = int(round(split_ratio * len(files)))
train_files = set(random.sample(files, num_train))
test_files = set(files) - train_files
```

- **Usage**:
```bash
python dataset_manager.py -i ./segmented -o ./data_split -s 0.8
```

Organizes final data into:
```
data_split/
    ├── train/
    │   ├── class0/ # audio data
    │   └── class1/ # audio data
    └── test/
        ├── class0/ # audio data
        └── class1/ # audio data
```

---

### 6. File Manager (Overlap Checker) (`file_manager.py`)

**Purpose**: Prevents data leakage by ensuring no overlap of segments from the same source file in both train and test sets.

- **Core Overlap Check**:
```python
def extract_group_key(filename):
    if "_" in filename:
        return filename.split("_")[0]
    return os.path.splitext(filename)[0]
```

- **Usage**:
```bash
python file_manager.py -i ./data_split --fix
```

Moves or removes duplicates from minority sets to maintain a strict separation.

---

### 7. Submodel Trainer (`submodel_trainer.py`)

**Purpose**: Trains individual binary classifiers (Real vs. **one** Synthetic type) using mel-spectrogram input.

- **Key Points**:
  1. **Mel-Spectrogram Generation** using `torchaudio.transforms.MelSpectrogram`.
  2. **Model**: ResNet backbone from TIMM, final FC layer outputs 2 logits (`[Real, Synthetic]`).
  3. **Loss**: Cross-entropy for binary classification.

---

<details>
<summary>Code Snippet</summary>

```python
model = timm.create_model(args.model_name, pretrained=False, num_classes=0)
# Optionally freeze base.
# Then attach a 2-output classification head.
model.head = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(model.num_features, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 2)  # [Real, Synthetic]
)
```

</details>

- **Usage**:
```bash
python submodel_trainer.py --data-dir ./data_split --model-name resnet18 --epochs 30
```

**Note**: In practice, you may train multiple sub-models, each specialized for detecting a different synthetic category, or simply train repeated times with different random seeds. This yields diverse classifiers that we later merge.

#### Visual Explanation: Training Multiple Sub-models

Imagine you have different synthetic classes, like `SyntheticA, SyntheticB, SyntheticC`. You can produce separate labeled datasets (each marking real vs. a specific synthetic type), or you can filter your data to a binary partition each time. Each sub-model is trained to distinguish Real from one type (or a subset of synthetic types). You end up with multiple `.pth` files:

```
┌───────────────────────────────┐
│ [DATA: Real vs SyntheticA]    │
│                               │
│ Submodel_A.pth                │
└───────────────────────────────┘
┌───────────────────────────────┐
│ [DATA: Real vs SyntheticB]    │
│                               │
│ Submodel_B.pth                │
└───────────────────────────────┘
...
┌───────────────────────────────┐
│ [DATA: Real vs SyntheticN]    │
│                               │
│ Submodel_N.pth                │
└───────────────────────────────┘
```

This approach leverages separate training runs to learn different decision boundaries, improving robustness once aggregated.

---

### 8. Model Merger (`model_merger.py`)

**Purpose**: Loads multiple trained 2-logit sub-models (each `[Real, Synthetic]`) and merges them into an ensemble model that outputs `[syn1, syn2, ..., synN, real_averaged]`.

- **Core Merging Excerpt**:
```python
sub_models = []
for idx, entry in enumerate(submodel_entries, start=1):
    sm = load_sub_model(model_path, device, model_name=args.model_name)
    sub_models.append(sm)

merged = ModularMultiHeadClassifier(sub_models)
# This class's forward() concatenates synthetic logits and averages real logits.
```

- **Usage**:
```bash
python model_merger.py --submodels-folder ./trained_models \
  --csv-file submodels.csv --output-path merged_model.pth
```

In a typical `submodels.csv`:
```
model_filename,synthetic_class,real_class
submodelA.pth,SyntheticA,Real
submodelB.pth,SyntheticB,Real
submodelC.pth,SyntheticC,Real
```

#### Visual Explanation: Aggregating Sub-models

Below is an ASCII depiction of how the merger combines the multiple binary heads. Each sub-model’s `[Real, Synthetic]` logits are separated into `zX_real` and `zX_syn`. All `zX_syn` are kept distinct, while the real logits are averaged:

```
Submodel A -> ( zA_real , zA_syn )
Submodel B -> ( zB_real , zB_syn )
Submodel C -> ( zC_real , zC_syn )

                [   zA_syn    zB_syn    zC_syn   average( zA_real , zB_real , zC_real ) ]
                                    ↓
Final Ensemble Output => [ synA , synB , synC , real_averaged ]
```

**Why Average the Real Logits?**

- The system requires *all* sub-models to be confident the audio is real to achieve a high `real_averaged` score.
- If *any* sub-model strongly suspects synthetic, its synthetic logit can exceed the real average, tilting the final classification to *fake*.

**Merging** thus produces an integrated model that can detect multiple synthetic classes while maintaining a single *Real* output, making inference simpler.


**Purpose**: Loads multiple trained 2-logit sub-models and merges them into an ensemble model that outputs `[syn1, syn2, ..., synN, real_averaged]`.

- **Core Merging Excerpt**:
```python
sub_models = []
for idx, entry in enumerate(submodel_entries, start=1):
    sm = load_sub_model(model_path, device, model_name=args.model_name)
    sub_models.append(sm)

merged = ModularMultiHeadClassifier(sub_models)
# This class's forward() concatenates synthetic logits and averages real logits.
```

- **Usage**:
```bash
python model_merger.py --submodels-folder ./trained_models \
  --csv-file submodels.csv --output-path merged_model.pth
```

**`submodels.csv`** typically has columns like:
```
model_filename,synthetic_class,real_class
modelA.pth,Synthetic1,Real
modelB.pth,Synthetic2,Real
...
```

---

### 9. Inference Runner (`inference_runner.py`)

**Purpose**: Applies the merged model to new audio, automatically handling segmentation, spectrogram creation, model inference, and final JSON output.

- **Core Logic**:
```python
with torch.no_grad():
    outputs = model(spec_batch)  # => [batch_size, N+1]
    # interpret: columns 0..N-1 => synthetic heads, last => real_averaged
    # Decide label by comparing max synthetic vs real.
```

- **Usage**:
```bash
python inference_runner.py \
  --merged-model merged_model.pth \
  --audio newAudio.wav \
  --output-json prediction.json
```

The script processes overlapping 4s windows from `newAudio.wav`, infers each chunk, and aggregates results.

---

## Sample Inference Output

An example final JSON:
```json
{
  "filename": "./newAudio.wav",
  "segments": [
    {"start_sec": 0.0, "end_sec": 4.0, "label": "SyntheticOne", "confidence": 0.93},
    {"start_sec": 4.0, "end_sec": 8.0, "label": "Real", "confidence": 0.89},
    {"start_sec": 8.0, "end_sec": 12.0, "label": "SyntheticTwo", "confidence": 0.95}
  ],
  "percentages": {
    "SyntheticOne": 30.0,
    "SyntheticTwo": 45.0,
    "Real": 25.0
  }
}
```

- **segments**: Per-chunk classification.
- **percentages**: Overall proportion of each predicted class.

---

## Code Requirements

### Prerequisites

Ensure you have the following installed:

- **CUDA 11.8 | 12.4 |s 12.6**
- **Python 3.10**
- **PyTorch - pip distro**: [pytorch.org](https://pytorch.org/)
- **torchaudio**: [pytorch.org/audio](https://pytorch.org/audio/stable/index.html)
- **timm**: [github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
- **Librosa**: [librosa.org](https://librosa.org/)
- **tqdm**: [github.com/tqdm/tqdm](https://github.com/tqdm/tqdm)
- **ffmpeg**: [ffmpeg.org](https://ffmpeg.org/)

### Python Packages

Install the following Python packages:

```bash
pip install librosa>=0.8.0 soundfile>=0.10.3.post1 scipy>=1.7.0
pip install numpy>=1.21.0 pandas>=1.3.0
pip install tqdm>=4.62.0
pip install torch>=1.10.0 torchaudio>=0.10.0 torchvision>=0.11.0 tensorboard>=2.8.0
pip install timm>=0.4.12
```

### Summary

- **Audio processing & signal analysis**:
    - `librosa>=0.8.0`
    - `soundfile>=0.10.3.post1`
    - `scipy>=1.7.0`
- **Data manipulation**:
    - `numpy>=1.21.0`
    - `pandas>=1.3.0`
- **Progress bar**:
    - `tqdm>=4.62.0`
- **PyTorch ecosystem**:
    - `torch>=1.10.0`
    - `torchaudio>=0.10.0`
    - `torchvision>=0.11.0`
    - `tensorboard>=2.8.0`
- **Model architectures**:
    - `timm>=0.4.12`

Installation:
```bash
pip install torch torchaudio timm librosa tqdm
```

---

## License

MIT License – see [LICENSE](LICENSE) for details.

---

##### For more in-depth mathematical treatment and performance analysis, refer to the accompanying **whitepaper**.

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
- Python 3.10
- PyTorch
- FFmpeg (for audio processing)
- Librosa (for feature extraction)
- NumPy, Pandas, tqdm, timm, scikit-learn, scipy (for data processing)

### Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/TtesseractT/Synthetic-Audio-Detection.git
   cd Synthetic-Audio-Detection
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
python audio_convert.py --input-dir path/to/data --output-dir path/to/converted
```
- **Arguments**:
    - `-i`, `--input`: Path to the input directory containing audio files (required).
    - `-o`, `--output`: Path to the output directory where converted audio files will be saved (required).
Normalizes audio format and sampling rate.

#### Audio Augmentation
```sh
python audio_augment.py --input-dir path/to/converted --output-dir path/to/augmented
```
- **Arguments**:
    - `-i`, `--input`: Input file or folder (required).
    - `-o`, `--output`: Output folder (required).
    - `-c`, `--csv`: CSV output file path (optional).
    - `-p`, `--pool-size`: Number of processes in the pool (default: number of CPU cores).
Generates augmented audio examples for improved training.

#### Audio Segmentation
```sh
python audio_segmenter.py --input-dir path/to/augmented --output-dir path/to/segmented --segment-duration 4
```
- **`--input-dir`**: Directory containing the augmented audio files.
- **`--output-dir`**: Directory to save the segmented audio files.
- **`--segment-duration`**: Duration of each audio segment in seconds (default: 4 seconds).

Splits audio files into fixed-length segments.

#### Dataset Management
```sh
python dataset_manager.py --input-dir path/to/segmented --output-dir path/to/dataset_split --split 0.8 --threads 4
```
- **`--input-dir`**: Input directory containing class folders with audio files.
- **`--output-dir`**: Output directory to store split audio files (with train/test subdirectories).
- **`--split`**: Split ratio for train (e.g., 0.8 means 80% train, 20% test; default: 0.5).
- **`--threads`**: Number of threads to use (default: 1).

Splits data into training and testing sets.

#### Overlap Checking (Logging Mixed data across classes)
```sh
python file_manager.py --input-dir path/to/dataset_split
```
- **`--input-dir`**: Base directory containing 'train' and 'test' subdirectories with class folders.

#### Optional --fix broken audio directories (Cleanup)
```sh
python file_manager.py --input-dir path/to/dataset_split --fix
```
- **`--fix`**: If provided, move files from the minority side into the majority side for each overlapping group.

Ensures no data leakage between training and testing sets.

### 2. Model Training
Train the individual sub-models using multiple GPUs and various arguments:

```sh
python submodel_trainer.py --data-dir path/to/dataset_split/train \
    --batch-size 32 \
    --epochs 50 \
    --lr 0.001 \
    --workers 20 \
    --seed 42 \
    --gpu 0 \
    --num_gpus 2 \
    --checkpoint-dir ./checkpoints \
    --resume path/to/checkpoint.pth \
    --Class0 Real \
    --Class1 Synthetic \
    --evaluate
```

- **`--data-dir`**: Path to the dataset.
- **`--batch-size`**: Batch size per GPU.
- **`--epochs`**: Number of total epochs to run.
- **`--lr`**: Initial learning rate.
- **`--workers`**: Number of data loading workers.
- **`--seed`**: Seed for initializing training.
- **`--gpu`**: GPU id to use.
- **`--num_gpus`**: Number of GPUs to use.
- **`--checkpoint-dir`**: Directory to save checkpoints.
- **`--resume`**: Path to resume checkpoint.
- **`--evaluate`**: Evaluate model on validation set.
- **`--Class0`**: Name of Class 0 (e.g., Real).
- **`--Class1`**: Name of Class 1 (e.g., Synthetic).

### 3. Model Merging
Merge trained sub-models into a unified multi-head model:
```sh
python model_merger.py --submodels-folder path/to/trained_models --csv-file submodels.csv --output-path path/to/merged_model.pth
```
- **`--submodels-folder`**: Folder containing sub-model `.pth` files.
- **`--csv-file`**: CSV file with columns "model_filename", "synthetic_class", and "real_class".
- **`--model-name`**: Name of the model architecture (default: 'resnet18').
- **`--output-path`**: Path to save the merged model `.pth` file.

### 4. Inference
Run the trained model on new audio data:
```sh
python inference_runner.py --merged-model path/to/merged_model.pth --audio path/to/audio --threshold 0.5 --device cuda --confidence-threshold 0.45 --smooth --output-json results.json
```
- **`--merged-model`**: Path to the merged model `.pth` file.
- **`--audio`**: Path to the input WAV file.
- **`--threshold`**: Threshold for deciding Real vs Synthetic (default: 0.5).
- **`--device`**: Device to run the inference on (default: "cuda").
- **`--confidence-threshold`**: Confidence threshold for segments (default: 0.45).
- **`--smooth`**: Apply smoothing across windows (optional).
- **`--output-json`**: Path to save the output JSON file (default: "results.json").

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

```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
