
# Audio Classification Training Script

This repository contains a training script for an audio classification task. The script loads audio files, converts them into spectrograms, applies data augmentation, and trains a deep learning model (using a pretrained ResNet from the [timm](https://github.com/rwightman/pytorch-image-models) library) with a custom classification head for 5 classes. It also includes evaluation routines and logging integration with TensorBoard.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)
- [Configuration Options](#configuration-options)
- [TensorBoard Integration](#tensorboard-integration)
- [License](#license)
- [Author](#author)

---

## Overview

This training script is designed to perform audio classification using deep learning. It processes raw `.wav` files by converting them into mel spectrograms, applies data augmentation for training, and then uses a model from the timm library that is fine-tuned with a custom classification head. The script supports multi-GPU training via `DataParallel` and logs the training progress to both file and TensorBoard.

---

## Features

- **Custom Dataset**: Reads audio files, converts them into mel spectrograms, and applies augmentations.
- **Flexible Model Selection**: Uses timm to list and create various ResNet models.
- **Training and Validation**: Implements both training and validation loops with progress tracking via tqdm.
- **Checkpointing**: Saves model checkpoints and supports resuming from a checkpoint.
- **Learning Rate Scheduling**: Uses ReduceLROnPlateau for adaptive learning rate adjustment.
- **TensorBoard Logging**: Logs loss, accuracy, and learning rate for visualization.
- **Evaluation Metrics**: Calculates per-class accuracy, confusion matrix, and classification report using scikit-learn.

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

---

## Usage

To run the training script, execute:

```bash
python train.py --data-dir ./dataset --epochs 30 --batch-size 32 --lr 0.0001 --gpu 0 --num_gpus 1
```

For evaluation after training, add the `--evaluate` flag:

```bash
python train.py --data-dir ./dataset --evaluate
```

You can also resume training from a checkpoint by specifying the `--resume` argument with the checkpoint path.

---

## Configuration Options

The script accepts several command-line arguments. Key options include:

- `--data-dir`: Path to the dataset directory.
- `--batch-size`: Batch size per GPU.
- `--epochs`: Total number of epochs.
- `--lr`: Initial learning rate.
- `--workers`: Number of data loader workers.
- `--seed`: Random seed for reproducibility.
- `--gpu`: GPU id to use.
- `--num_gpus`: Number of GPUs.
- `--checkpoint-dir`: Directory to save checkpoints.
- `--resume`: Path to a checkpoint to resume training.
- `--evaluate`: Flag to run evaluation after training.
- `--model-name`: Model name from timm (uses a list of ResNet models).

---

## Dataset Structure

The script expects the dataset to be organized in the following structure:

```
dataset/
  ├── train/
  │     ├── Real/
  │     ├── class1/
  │     ├── class2/
  │     ├── class3/
  │     └── class4/
  └── test/
        ├── Real/
        ├── class1/
        ├── class2/
        ├── class3/
        └── class4/
```

---

## TensorBoard Integration

TensorBoard is used to monitor training progress. The script logs step-level and epoch-level metrics:

```bash
tensorboard --logdir=runs
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
