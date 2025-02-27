# 🎵 AI-Powered Audio Classification: Real vs Synthetic Detection

![Audio Analysis](https://source.unsplash.com/1600x400/?sound,technology)

## 📌 Overview

This repository provides a **high-performance pipeline** for detecting whether an audio file is **real or AI-generated**. The system processes, augments, trains, and classifies audio files using **deep learning** and **spectrogram analysis**, leveraging **PyTorch, torchaudio, librosa, and timm**.

🚀 **Key Features:**
- **🔄 Audio Augmentation** – Enhance datasets with speed changes, pitch shifts, noise injection, and filtering.
- **🎛️ Format Conversion & Segmentation** – Standardize audio files and split them into fixed-length segments.
- **🧠 Deep Learning Model** – Train and merge multiple classifiers for robust inference.
- **📊 Probability-Based Inference** – Process overlapping windows and smooth classification results.

---

## 📂 Project Structure

```plaintext
.
├── audio_augmneter_batch.py      # Augments audio data with various transformations
├── audio_convert_segmenter.py    # Splits audio files into 4-sec mono segments
├── audio_convert_wave.py         # Converts audio to WAV format (32kHz, mono, 16-bit PCM)
├── file_rename_hash.py           # Renames files using SHA256 hash values
├── train_submodel.py             # Trains individual binary classifiers on spectrograms
├── merge_model_classifier.py     # Merges multiple sub-models into a unified classifier
├── inference_classifier.py       # Runs inference using the merged classifier model

```

---

## 📦 Installation

### ✅ Prerequisites
- **Python 3.10**
- **PyTorch + torchvision + torchaudio**
- **librosa**
- **timm (Torch Image Models)**
- **ffmpeg**

### ⚡ Setup
```bash
# Clone the repository
git clone https://github.com/TtesseractT/Synthetic-Audio-Detection.git
cd Synthetic-Audio-Detection

# Install dependencies
pip install -r requirements.txt
```

---

## 🎯 Usage

### 🔄 **Audio Augmentation**
```bash
python audio_augmneter_batch.py -i <input_folder> -o <output_folder>
```

### 🎛 **Audio Conversion & Segmentation**
```bash
python audio_convert_wave.py -i <input_folder> -o <output_folder>
python audio_convert_segmenter.py -i <input_folder> -o <output_folder>
```

### 🔑 **File Renaming (SHA256-based)**
```bash
python file_rename_hash.py -i <directory> -r  # -r for recursive mode
```

### 🏋️ **Training a Sub-Model**
```bash
python train_submodel.py --data-dir <dataset_directory> --epochs 100 --batch-size 32
```

### 🔗 **Merging Sub-Models into One Classifier**
Prepare a CSV file with columns `model_filename`, `synthetic_class`, and `real_class`:
```bash
python merge_model_classifier.py --submodels-folder <folder> --csv-file <csv_file> --output-path merged_model.pth
```

### 🎯 **Inference on New Audio Files**
```bash
python inference_classifier.py --merged-model merged_model.pth --audio <audio_file.wav> --threshold 0.5 --smooth
```

---

## 🏆 Model Training & Architecture

📊 **Training Workflow:**
1. **Preprocess audio files** → Convert to spectrograms
2. **Train binary classifiers** → Identify real vs synthetic samples
3. **Merge sub-models** → Create a single ensemble classifier
4. **Perform inference** → Classify new audio samples

🛠 **Architecture Highlights:**
- **ResNet-based classifier** (pretrained `timm` models)
- **Binary classification per sub-model** (Real vs Synthetic)
- **Multi-head merged classifier** for robust predictions

---

## 📈 Example Output
📌 Sample classification output in `results.json`:
```json
{
    "filename": "audio_sample.wav",
    "segments": [
        {"start_sec": 0.0, "end_sec": 4.0, "label": "SyntheticOne"},
        {"start_sec": 4.0, "end_sec": 8.0, "label": "Real"}
    ],
    "percentages": {
        "SyntheticOne": 50.24,
        "Real": 48.02
    }
}
```

---

## 🤝 Contributing
We welcome contributions! Please submit a pull request or open an issue for suggestions, bug fixes, or feature requests.

---

## 📜 License
This project is licensed under the **MIT License**.

---

## 📚 Citations & References
This project leverages open-source frameworks:
- **PyTorch**: https://pytorch.org/
- **torchaudio**: https://pytorch.org/audio/
- **librosa**: https://librosa.org/
- **timm (Torch Image Models)**: https://github.com/rwightman/pytorch-image-models

*Developed with ❤️ by [Your Name / Organization]* 🚀

