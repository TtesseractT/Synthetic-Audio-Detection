#!/usr/bin/env python3
# Company:  Uhmbrella Ltd 2025
# Author:   Sabian Hibbs
# Date:     2025-01-01
# Version:  1.0
# License:  MIT


import torch
import torchaudio
import torchvision.transforms as transforms
import timm
import time
import torch.nn as nn
import torch.nn.functional as F
import argparse
import traceback
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, List, Dict
from dataclasses import dataclass
import sys
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d
import random
import json
import concurrent.futures


seed = 9
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class AudioConfig:
    """Configuration for audio processing parameters"""
    target_sample_rate: int = 32000
    window_size: float = 4.0  # seconds
    overlap: float = 0.85  
    normalize_audio: bool = True


@dataclass
class SpectrogramConfig:
    """Configuration for spectrogram transformation"""
    sample_rate: int = 32000
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    f_min: int = 20
    f_max: int = 12000
    top_db: int = 80
    power: float = 2.0
    norm: str = 'slaney'


class AudioAnalyzer:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.classes = ['Class1', 'Class2', 'Class3', 'Class4', 'Class5']
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.audio_config = AudioConfig()
        self.spec_config = SpectrogramConfig()

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.spec_config.sample_rate,
            n_fft=self.spec_config.n_fft,
            hop_length=self.spec_config.hop_length,
            n_mels=self.spec_config.n_mels,
            f_min=self.spec_config.f_min,
            f_max=self.spec_config.f_max,
            power=self.spec_config.power,
            norm=self.spec_config.norm
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=self.spec_config.top_db)

        
        self.sensitivity_factors = {
            'class1': 1.0,
            'class2': 1.0,
            'class3': 1.0,
            'class4': 1.0,
            'class5': 1.0
        }
        self.confidence_threshold = 0.45

        self.model = self._load_model(model_path)

    def _load_model(self, checkpoint_path: str) -> nn.Module:
        model = timm.create_model('resnet152', pretrained=False, num_classes=0)
        model.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(model.num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, len(self.classes))  # Expected: len(self.classes) == 5
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Remove keys for the final layer if their shapes don't match.
        for key in ['head.10.weight', 'head.10.bias']:
            if key in state_dict:
                if key == 'head.10.weight' and state_dict[key].shape != model.head[-1].weight.shape:
                    print(f"Removing mismatched key {key} from state_dict")
                    del state_dict[key]
                elif key == 'head.10.bias' and state_dict[key].shape != model.head[-1].bias.shape:
                    print(f"Removing mismatched key {key} from state_dict")
                    del state_dict[key]
        
        # Load the rest of the state dict.
        model.load_state_dict(state_dict, strict=False)
        
        # Reinitialize the final layer so it can learn from scratch.
        nn.init.kaiming_normal_(model.head[-1].weight, nonlinearity='linear')
        if model.head[-1].bias is not None:
            nn.init.constant_(model.head[-1].bias, 0)
        
        model.to(self.device).eval()
        return model


    def normalize_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = waveform - waveform.mean()
        peak = torch.abs(waveform).max()
        if peak > 0:
            waveform = waveform / peak
        rms = torch.sqrt(torch.mean(waveform ** 2))
        target_rms = 0.2
        if rms > 0:
            waveform = waveform * (target_rms / rms)
        return waveform

    def apply_noise_reduction(self, waveform: torch.Tensor) -> torch.Tensor:
        # omitted details for brevity
        return waveform  # implement your noise reduction if needed

    def preprocess_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0)

        if sample_rate != self.audio_config.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.audio_config.target_sample_rate
            )
            waveform = resampler(waveform)
            sample_rate = self.audio_config.target_sample_rate

        # Pad if < 4s
        audio_length_seconds = waveform.size(0) / sample_rate
        if audio_length_seconds < 4.0:
            desired_len_samples = int(5.0 * sample_rate)
            padded_waveform = torch.zeros(desired_len_samples, dtype=waveform.dtype)
            padded_waveform[:waveform.size(0)] = waveform
            waveform = padded_waveform

        # Optional normalization & noise reduction
        if self.audio_config.normalize_audio:
            waveform = self.normalize_audio(waveform)
            waveform = self.apply_noise_reduction(waveform)

        return waveform, sample_rate

    # -------------------- SPECTROGRAM --------------------
    def process_window(self, window: torch.Tensor) -> torch.Tensor:
        window = window.unsqueeze(0)  # => (1, samples)
        spec = self.mel_spec(window)
        spec = self.amplitude_to_db(spec)
        # mean-std normalize
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        spec = transforms.Resize((512, 512))(spec)
        spec = spec.repeat(1, 3, 1, 1)  # => (1, 3, 512, 512)
        return spec  # shape => (1, 3, 512, 512)

    def adjust_probabilities(self, probs: np.ndarray) -> np.ndarray:
        adjusted = probs.copy()
        for idx, cls in enumerate(self.classes):
            adjusted[idx] *= self.sensitivity_factors[cls]
        return adjusted / adjusted.sum()

    def smooth_predictions(self, predictions: List[int], probabilities: List[np.ndarray]) -> Tuple[List[int], List[np.ndarray]]:
        if not probabilities:
            return [], []
        preds_array = np.array(predictions)
        probs_array = np.array(probabilities)

        smoothed_probs = np.zeros_like(probs_array)
        for i in range(probs_array.shape[1]):
            smoothed_probs[:, i] = gaussian_filter1d(probs_array[:, i], sigma=2)

        smoothed_probs = smoothed_probs / smoothed_probs.sum(axis=1, keepdims=True)
        smoothed_preds = np.argmax(smoothed_probs, axis=1)
        final_preds = medfilt(smoothed_preds, kernel_size=5)

        max_probs = smoothed_probs.max(axis=1)
        confident_mask = (max_probs >= self.confidence_threshold)
        if len(final_preds) > 0:
            majority_class = np.argmax(np.bincount(final_preds))
            final_preds[~confident_mask] = majority_class

        return final_preds.tolist(), smoothed_probs.tolist()

    def get_confident_segments(self, timestamps, predictions, probabilities) -> List[Dict]:
        segments = []
        if not predictions:
            return segments
        idx = 0
        while idx < len(predictions):
            current_class = predictions[idx]
            start_idx = idx
            while (idx + 1 < len(predictions) and predictions[idx + 1] == current_class):
                idx += 1
            end_idx = idx
            segment_probs = [probabilities[i][current_class] for i in range(start_idx, end_idx + 1)]
            avg_confidence = float(np.mean(segment_probs))
            if avg_confidence >= self.confidence_threshold:
                seg_start = float(timestamps[start_idx])
                seg_end = float(timestamps[end_idx] + self.audio_config.window_size)
                segments.append({
                    "start": seg_start,
                    "end": seg_end,
                    "class": self.classes[current_class],
                    "confidence": avg_confidence
                })
            idx += 1
        return segments

    def analyze_audio(self, audio_path: str) -> Dict:
        waveform, sample_rate = self.preprocess_audio(audio_path)
        window_samples = int(self.audio_config.window_size * sample_rate)
        hop_samples = int((1 - self.audio_config.overlap) * window_samples)
        silence_threshold = 1e-4

        # Collect windows
        windows = []
        timestamps = []
        for start_idx in range(0, waveform.size(0) - window_samples + 1, hop_samples):
            window_chunk = waveform[start_idx:start_idx + window_samples]
            if window_chunk.abs().max() < silence_threshold:
                continue
            windows.append(window_chunk)
            timestamps.append(start_idx / sample_rate)

        if not windows:
            return {
                "percentages": {cls: 0.0 for cls in self.classes},
                "segments": []
            }

        # Build a list of spectrograms
        specs_list = []
        for w in windows:
            spec = self.process_window(w)  # => (1, 3, 512, 512)
            specs_list.append(spec)

        # (num_windows, 3, 512, 512)
        all_specs = torch.cat(specs_list, dim=0).to(self.device)

        # Inference with mini-batches + mixed precision
        batch_size = 256
        outputs_list = []
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for i in range(0, all_specs.size(0), batch_size):
                batch_specs = all_specs[i:i+batch_size]
                out = self.model(batch_specs)
                outputs_list.append(out.float())  # ensure final is float32 for stable post-processing

        outputs = torch.cat(outputs_list, dim=0)  # => (num_windows, 5)
        probabilities = []
        predictions = []
        for out in outputs:
            probs = F.softmax(out, dim=0).cpu().numpy()
            adjusted_probs = self.adjust_probabilities(probs)
            probabilities.append(adjusted_probs)
            predictions.append(np.argmax(adjusted_probs))

        smoothed_preds, smoothed_probs = self.smooth_predictions(predictions, probabilities)
        if len(smoothed_probs) > 0:
            final_probs = np.mean(smoothed_probs, axis=0)
        else:
            final_probs = np.zeros(len(self.classes), dtype=np.float32)

        percentages = {cls: float(p * 100.0) for cls, p in zip(self.classes, final_probs)}
        segments = self.get_confident_segments(timestamps, smoothed_preds, smoothed_probs)

        return {
            "percentages": percentages,
            "segments": segments
        }


def parallel_analyze(analyzer: AudioAnalyzer, audio_files: List[Path]) -> List[Dict]:
    """
    Example parallel pipeline:
     - In parallel: preprocess the audio files on CPU.
     - Then on main process: run GPU inference with half/mixed precision.
    """
    results_list = []
    def cpu_preproc(file_path: Path):
        waveform, sr = analyzer.preprocess_audio(str(file_path))
        return (waveform, sr, file_path.name)

    preprocessed_data = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        future_to_file = {executor.submit(cpu_preproc, f): f for f in audio_files}
        for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(audio_files), desc="Preprocessing"):
            fpath = future_to_file[future]
            try:
                waveform, sr, fname = future.result()
                preprocessed_data.append((waveform, sr, fname))
            except Exception as e:
                print(f"Error preprocessing {fpath}: {e}")

    # GPU inference in main process
    for (waveform, sr, fname) in tqdm(preprocessed_data, desc="Inference"):
        analysis_result = analyze_waveform(analyzer, waveform, sr, fname)
        results_list.append(analysis_result)

    return results_list


def analyze_waveform(analyzer: AudioAnalyzer, waveform: torch.Tensor, sample_rate: int, fname: str) -> Dict:
    """
    If we want to replicate the logic of `analyze_audio`, but start from a raw waveform:
    We'll do window slicing, spectrogram, and GPU inference with autocast.
    """
    window_samples = int(analyzer.audio_config.window_size * sample_rate)
    hop_samples = int((1 - analyzer.audio_config.overlap) * window_samples)
    silence_threshold = 1e-4

    windows = []
    timestamps = []
    for start_idx in range(0, waveform.size(0) - window_samples + 1, hop_samples):
        window_chunk = waveform[start_idx:start_idx + window_samples]
        if window_chunk.abs().max() < silence_threshold:
            continue
        windows.append(window_chunk)
        timestamps.append(start_idx / sample_rate)

    if not windows:
        return {
            "filename": fname,
            "percentages": {cls: 0.0 for cls in analyzer.classes},
            "segments": []
        }

    specs_list = []
    for w in windows:
        spec = analyzer.process_window(w)
        specs_list.append(spec)

    all_specs = torch.cat(specs_list, dim=0).to(analyzer.device)

    # Mixed-precision inference
    batch_size = 256
    outputs_list = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for i in range(0, all_specs.size(0), batch_size):
            mini_batch = all_specs[i:i+batch_size]
            out = analyzer.model(mini_batch)
            outputs_list.append(out.float())

    outputs = torch.cat(outputs_list, dim=0)
    probabilities = []
    predictions = []
    for out in outputs:
        probs = F.softmax(out, dim=0).cpu().numpy()
        adjusted = analyzer.adjust_probabilities(probs)
        probabilities.append(adjusted)
        predictions.append(np.argmax(adjusted))

    smoothed_preds, smoothed_probs = analyzer.smooth_predictions(predictions, probabilities)
    if len(smoothed_probs) > 0:
        final_probs = np.mean(smoothed_probs, axis=0)
    else:
        final_probs = np.zeros(len(analyzer.classes), dtype=np.float32)

    percentages = {cls: float(p * 100.0) for cls, p in zip(analyzer.classes, final_probs)}
    segments = analyzer.get_confident_segments(timestamps, smoothed_preds, smoothed_probs)
    return {
        "filename": fname,
        "percentages": percentages,
        "segments": segments
    }


def main():
    parser = argparse.ArgumentParser(description='5-Class Audio Inference with Mixed Precision.')
    parser.add_argument('--audio_path', type=str, help='Path to single audio file')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Model checkpoint path (must be for 5 classes)')
    parser.add_argument('--output_dir', type=str, default='results_json')
    parser.add_argument('--confidence_threshold', type=float, default=0.45)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--IsBatch', type=str, default=None, help='Folder for batch mode')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel CPU preprocessing')

    args = parser.parse_args()
    try:
        ckpt = Path(args.checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        analyzer = AudioAnalyzer(str(ckpt), device=args.device)
        analyzer.confidence_threshold = args.confidence_threshold

        output_folder = Path(args.output_dir)
        output_folder.mkdir(parents=True, exist_ok=True)
        json_path = output_folder / "results.json"

        results_list = []

        if args.IsBatch:
            batch_folder = Path(args.IsBatch)
            if not batch_folder.exists() or not batch_folder.is_dir():
                raise NotADirectoryError(f"Batch folder not found: {batch_folder}")

            all_audio_files = sorted(batch_folder.glob("*.*"))
            if not all_audio_files:
                print("No files found in batch folder.")
            else:
                if args.parallel:
                    print("Parallel CPU Preprocessing + Mixed Precision GPU Inference")
                    results_list = parallel_analyze(analyzer, all_audio_files)
                else:
                    for audio_file in tqdm(all_audio_files, desc="Processing"):
                        if audio_file.is_dir():
                            continue
                        try:
                            analysis_result = analyzer.analyze_audio(str(audio_file))
                            file_dict = {"filename": audio_file.name}
                            file_dict.update({cls: f"{p:.3f}" for cls, p in analysis_result["percentages"].items()})
                            file_dict["segments"] = analysis_result["segments"]
                            results_list.append(file_dict)
                        except Exception as e:
                            print(f"Skipping file {audio_file} due to error: {e}")
        else:
            # Single file mode
            audio_path = Path(args.audio_path)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            analysis_result = analyzer.analyze_audio(str(audio_path))
            file_dict = {"filename": audio_path.name}
            file_dict.update({cls: f"{p:.3f}" for cls, p in analysis_result["percentages"].items()})
            file_dict["segments"] = analysis_result["segments"]
            results_list.append(file_dict)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_list, f, indent=4)

        #print(f"Results written to: {json_path}")

    except Exception as e:
        print("Error:", e)
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
