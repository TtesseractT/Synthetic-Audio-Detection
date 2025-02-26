#!/usr/bin/env python3
import os
import sys
import json
import argparse
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

import torchaudio
import torchvision.transforms as transforms
import timm
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter1d

########################################################
# 1. Multi-Head Model: Each sub-model => [Real, Synthetic]
########################################################

class BinaryClassifier(nn.Module):
    """
    Sub-model with a backbone + 2-output head:
      index 0 => Real, index 1 => Synthetic
    """
    def __init__(self, model_name='resnet18'):
        super().__init__()
        self.base = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.base.num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # => [Real, Synthetic]
        )
    def forward(self, x):
        feats = self.base.forward_features(x)
        return self.head(feats)

class ModularMultiHeadClassifier(nn.Module):
    """
    The final merged model: has an nn.ModuleList of sub-models.
    On forward pass, it averages the Real outputs and keeps the Synthetic separate.
    => final output shape [B, N+1] where first N are synthetic and last is merged Real.
    """
    def __init__(self, sub_models: List[nn.Module]):
        super().__init__()
        self.sub_models = nn.ModuleList(sub_models)
    def forward(self, x):
        real_list = []
        syn_list = []
        for m in self.sub_models:
            out = m(x)  # => [B, 2]
            # Here we assume index 0 corresponds to Real and index 1 to Synthetic.
            real_list.append(out[:, 0:1])
            syn_list.append(out[:, 1:2])
        syn_cat = torch.cat(syn_list, dim=1)   # => [B, N]
        real_cat = torch.cat(real_list, dim=1)   # => [B, N]
        real_mean = torch.mean(real_cat, dim=1, keepdim=True)  # => [B, 1]
        return torch.cat([syn_cat, real_mean], dim=1)  # => [B, N+1]

########################################################
# 2. Loading the Merged Model from .pth (with metadata)
########################################################

def load_merged_model(merged_path: str, device: torch.device, backbone_name='resnet18'):
    """
    Loads the multi-head model from the saved checkpoint.
    Returns a tuple (final_model, metadata) where metadata is expected to contain the key "class_names".
    """
    state = torch.load(merged_path, map_location=device)
    sd = state['state_dict']
    metadata = state.get('metadata', None)
    if not metadata or "class_names" not in metadata:
        raise ValueError("Merged model checkpoint does not contain metadata for class names!")
    
    # Identify which sub-model indices exist: keys like "sub_models.<idx>.<...>"
    submodel_indices = set()
    for k in sd.keys():
        parts = k.split(".")
        if len(parts) >= 3 and parts[0] == "sub_models":
            try:
                idx = int(parts[1])
                submodel_indices.add(idx)
            except ValueError:
                pass
    submodel_indices = sorted(list(submodel_indices))
    print(f"Found {len(submodel_indices)} sub-model(s): {submodel_indices}")

    sub_models = []
    for idx in submodel_indices:
        sm = BinaryClassifier(model_name=backbone_name)
        local_sd = {}
        for param_key in sm.state_dict().keys():
            big_key = f"sub_models.{idx}." + param_key
            if big_key in sd:
                local_sd[param_key] = sd[big_key]
            else:
                local_sd[param_key] = sm.state_dict()[param_key]
        sm.load_state_dict(local_sd, strict=False)
        sm.to(device)
        sm.eval()
        sub_models.append(sm)

    final_model = ModularMultiHeadClassifier(sub_models)
    final_model.to(device).eval()

    # Quick test
    dummy_in = torch.randn(2, 3, 512, 512).to(device)
    dummy_out = final_model(dummy_in)
    print("Rebuilt merged model => dummy output shape:", dummy_out.shape)
    return final_model, metadata

########################################################
# 3. Overlapping Window + Spectrogram Logic
########################################################

@dataclass
class AudioConfig:
    sample_rate: int = 32000
    window_size: float = 4.0  # seconds
    overlap: float = 0.85     # fraction overlap
    silence_threshold: float = 1e-4

@dataclass
class SpectrogramConfig:
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    f_min: int = 20
    f_max: int = 12000
    top_db: int = 80
    norm: str = 'slaney'

def preprocess_waveform(path: str, cfg: AudioConfig):
    wf, sr = torchaudio.load(path)
    wf = wf.mean(dim=0)  # force mono
    if sr != cfg.sample_rate:
        wf = torchaudio.transforms.Resample(sr, cfg.sample_rate)(wf)
        sr = cfg.sample_rate
    needed = int(cfg.window_size * sr)
    if wf.shape[0] < needed:
        temp = torch.zeros(needed)
        temp[:wf.shape[0]] = wf
        wf = temp
    return wf, sr

def waveform_to_spectrogram(waveform: torch.Tensor, sr: int, spec_cfg: SpectrogramConfig):
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=spec_cfg.n_fft,
        hop_length=spec_cfg.hop_length,
        n_mels=spec_cfg.n_mels,
        f_min=spec_cfg.f_min,
        f_max=spec_cfg.f_max,
        norm=spec_cfg.norm
    )
    amp2db = torchaudio.transforms.AmplitudeToDB(top_db=spec_cfg.top_db)

    spec = mel(waveform.unsqueeze(0))  # => [1, n_mels, time]
    spec = amp2db(spec)
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    spec = transforms.Resize((512, 512))(spec)
    spec3 = spec.repeat(3, 1, 1)  # => [3,512,512]
    return spec3.unsqueeze(0)   # => [1,3,512,512]

def slice_waveform(wf: torch.Tensor, sr: int, cfg: AudioConfig):
    """
    Return a list of (chunk, timestamp).
    """
    window_samples = int(cfg.window_size * sr)
    hop_samples = int((1 - cfg.overlap) * window_samples)
    chunks = []
    timestamps = []
    for start_idx in range(0, wf.shape[0] - window_samples + 1, hop_samples):
        piece = wf[start_idx:start_idx + window_samples]
        if piece.abs().max() < cfg.silence_threshold:
            continue
        chunks.append(piece)
        timestamps.append(start_idx / sr)
    return chunks, timestamps

########################################################
# 4. Probability Interpretation (Using Metadata)
########################################################

def interpret_multihead_logits(logits: torch.Tensor, threshold=0.5, 
                               synthetic_names: List[str]=None, real_name: str = "Real"):
    """
    Given logits of shape [N_synthetic + 1] where the last index corresponds to the merged real output,
    decide the label. If the real probability is above the threshold and all synthetic outputs are below threshold,
    assign the real label; otherwise, assign the synthetic label with the highest probability.
    """
    s = torch.sigmoid(logits)  # shape [N+1]
    n = s.shape[0] - 1  # number of synthetic outputs
    syn_probs = s[:n]
    real_prob = s[-1]
    
    if real_prob >= threshold and (syn_probs < threshold).all():
        label = real_name
    else:
        idx = int(torch.argmax(syn_probs).item())
        if synthetic_names and idx < len(synthetic_names):
            label = synthetic_names[idx]
        else:
            label = f"Synthetic_{idx+1}"
    return label, s.cpu().numpy()

########################################################
# 5. Main Inference Logic
########################################################

def main():
    parser = argparse.ArgumentParser(
        description="Multi-head inference with overlapping windows using metadata from the merged model."
    )
    parser.add_argument('--merged-model', type=str, required=True, help="Path to merged .pth")
    parser.add_argument('--audio', type=str, required=True, help="Path to WAV file")
    parser.add_argument('--threshold', type=float, default=0.5, help="Threshold for deciding Real vs Synthetic")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--confidence-threshold', type=float, default=0.45, help="Confidence threshold for segments.")
    parser.add_argument('--smooth', action='store_true', help="Apply smoothing across windows.")
    parser.add_argument('--output-json', type=str, default="results.json")
    args = parser.parse_args()

    # For reproducibility
    seed = 9
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 1) Load merged model and extract metadata for class names.
    model, metadata = load_merged_model(args.merged_model, device)
    model.eval()

    # Expect metadata to contain "class_names" as a list: first N are synthetic, last is the real class.
    class_names = metadata["class_names"]
    synthetic_names = class_names[:-1]
    real_name = class_names[-1]
    print("Using metadata names:")
    print("Synthetic names:", synthetic_names)
    print("Real name:", real_name)

    # 2) Process audio: overlapping windows and spectrograms.
    audio_cfg = AudioConfig(sample_rate=32000, window_size=4.0, overlap=0.0, silence_threshold=1e-3)
    spec_cfg = SpectrogramConfig(n_fft=2048, hop_length=512, n_mels=128, f_min=20, f_max=12000, top_db=80, norm='slaney')

    wf, sr = preprocess_waveform(args.audio, audio_cfg)
    chunks, timestamps = slice_waveform(wf, sr, audio_cfg)

    if not chunks:
        print("No valid audio chunks found (all below silence threshold). Exiting.")
        out = {
            "filename": args.audio,
            "segments": [],
            "percentages": {},
        }
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=4)
        return

    # 3) Build spectrograms for each chunk and perform inference.
    specs = []
    for c in chunks:
        spec = waveform_to_spectrogram(c, sr, spec_cfg)
        specs.append(spec)
    all_specs = torch.cat(specs, dim=0).to(device)
    
    with torch.no_grad():
        outputs = []
        batch_size = 128
        for start in range(0, all_specs.shape[0], batch_size):
            mini = all_specs[start:start+batch_size]
            out = model(mini)  # => [b, N+1]
            outputs.append(out)
        outputs = torch.cat(outputs, dim=0)  # => [num_windows, N+1]

    # 4) Interpret outputs for each window.
    raw_labels = []
    raw_probs = []
    for row in outputs:
        label, s = interpret_multihead_logits(row, threshold=args.threshold,
                                              synthetic_names=synthetic_names, real_name=real_name)
        raw_labels.append(label)
        raw_probs.append(s)

    # Optionally apply smoothing.
    if args.smooth:
        raw_probs_arr = np.array(raw_probs)  # shape [num_windows, N+1]
        for dim in range(raw_probs_arr.shape[1]):
            raw_probs_arr[:, dim] = gaussian_filter1d(raw_probs_arr[:, dim], sigma=2)
        # Re-normalize each row.
        for i in range(raw_probs_arr.shape[0]):
            row_sum = raw_probs_arr[i].sum()
            if row_sum > 0:
                raw_probs_arr[i] /= row_sum
        # Pick final labels based on smoothed probabilities.
        smoothed_labels = []
        for i in range(raw_probs_arr.shape[0]):
            real_p = raw_probs_arr[i, -1]
            syn_p = raw_probs_arr[i, :-1]
            if real_p >= args.threshold and (syn_p < args.threshold).all():
                label2 = real_name
            else:
                idx = int(syn_p.argmax())
                if idx < len(synthetic_names):
                    label2 = synthetic_names[idx]
                else:
                    label2 = f"Synthetic_{idx+1}"
            smoothed_labels.append(label2)
        raw_labels = smoothed_labels
        raw_probs = raw_probs_arr.tolist()

    # 5) Summarize final probabilities by averaging.
    final_probs_arr = np.mean(raw_probs, axis=0)
    prob_dict = {}
    n_syn = len(final_probs_arr) - 1
    for i in range(n_syn):
        name = synthetic_names[i] if i < len(synthetic_names) else f"Synthetic_{i+1}"
        prob_dict[name] = float(final_probs_arr[i] * 100)
    prob_dict[real_name] = float(final_probs_arr[-1] * 100)

    # 6) Build segments with timestamps and labels.
    segments = []
    for i, lbl in enumerate(raw_labels):
        segments.append({
            "start_sec": timestamps[i],
            "end_sec": timestamps[i] + audio_cfg.window_size,
            "label": lbl
        })

    out_json = {
        "filename": args.audio,
        "segments": segments,
        "percentages": prob_dict
    }
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(out_json, f, indent=4)
    print("Wrote results to", args.output_json)
    print(json.dumps(out_json, indent=4))

if __name__=="__main__":
    main()
