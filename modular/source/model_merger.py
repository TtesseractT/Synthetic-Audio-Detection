#!/usr/bin/env python3
# Company:  Uhmbrella Ltd 2025
# Author:   Sabian Hibbs
# Date:     2025-01-01
# Version:  1.0
# License:  MIT


import os
import torch
import torch.nn as nn
import timm
import argparse
import copy
import csv
import collections

class BinaryClassifier(nn.Module):
    """
    Each sub-model outputs 2 logits: [Synthetic, Real].
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
            nn.Linear(256, 2)
        )
    def forward(self, x):
        feats = self.base.forward_features(x)
        return self.head(feats)

def force_separate_parameters(model: nn.Module):
    for name, param in model.named_parameters():
        param.data = param.data.clone()

def load_sub_model(checkpoint_path, device, model_name='resnet18'):
    """
    Load a 2-output model from 'checkpoint_path'.
    """
    model = BinaryClassifier(model_name=model_name)
    ck = torch.load(checkpoint_path, map_location=device)
    sd_in = ck['state_dict']

    # Direct load (since it's a 2-output model)
    model.load_state_dict(sd_in, strict=False)
    model.to(device)
    model.eval()
    force_separate_parameters(model)
    return copy.deepcopy(model)

class ModularMultiHeadClassifier(nn.Module):
    """
    Holds multiple sub-models, each outputting [Real, Synthetic] (Index 0: Real, Index 1: Synthetic).
    For each sub-model, the Synthetic output (Index 1) is kept individually and the
    Real outputs (Index 0) are averaged to form one final Real output.

    Final output shape is: [B, N+1] where:
      - Columns 0 to N-1 are synthetic outputs (one per sub-model).
      - Column N is the averaged real output.
    """
    def __init__(self, sub_models):
        super().__init__()
        self.sub_models = nn.ModuleList(sub_models)

    def forward(self, x):
        synthetic_list = []
        real_list = []
        for m in self.sub_models:
            out = m(x)  # Expected output shape: [B, 2], where index 0=Real, index 1=Synthetic

            # --- CORRECTED LOGIC ---
            # Collect the Real logit (at index 0)
            real_list.append(out[:, 0:1])
            # Collect the Synthetic logit (at index 1)
            synthetic_list.append(out[:, 1:2])
        synthetic_cat = torch.cat(synthetic_list, dim=1)           # [B, N] - Contains individual synthetic logits
        real_cat = torch.cat(real_list, dim=1)                     # [B, N] - Contains individual real logits
        real_mean = torch.mean(real_cat, dim=1, keepdim=True)      # [B, 1] - Contains the AVERAGE of real logits

        # Final output: Concatenate individual synthetic logits and the averaged real logit
        return torch.cat([synthetic_cat, real_mean], dim=1)        # [B, N+1]

def main():
    parser = argparse.ArgumentParser(
        description="Merge sub-models into a multi-head classifier with a merged Real output."
    )
    parser.add_argument('--submodels-folder', type=str, required=True,
                        help='Folder containing sub-model .pth files.')
    parser.add_argument('--csv-file', type=str, required=True,
                        help='CSV file with columns "model_filename", "synthetic_class", and "real_class".')
    parser.add_argument('--model-name', type=str, default='resnet18')
    parser.add_argument('--output-path', type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Read CSV file to get list of sub-models and their class names.
    submodel_entries = []
    with open(args.csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Each row should have keys: "model_filename", "synthetic_class", "real_class"
            submodel_entries.append(row)
    
    if not submodel_entries:
        print("No submodels found in CSV file!")
        return

    sub_models = []
    synthetic_names = []
    real_names = []
    # Process models in the order they appear in the CSV.
    for i, entry in enumerate(submodel_entries, start=1):
        model_file = entry['model_filename']
        syn_class = entry['synthetic_class']
        real_class = entry['real_class']
        model_path = os.path.join(args.submodels_folder, model_file)
        print(f"Loading sub-model {i} from {model_path} with synthetic class '{syn_class}' and real class '{real_class}'")
        sm = load_sub_model(model_path, device, model_name=args.model_name)
        sub_models.append(sm)
        synthetic_names.append(syn_class)
        real_names.append(real_class)

    merged = ModularMultiHeadClassifier(sub_models).to(device).eval()

    # Determine the final real class.
    if len(set(real_names)) == 1:
        merged_real_class = real_names[0]
    else:
        # If not all real_class values are identical, choose the most common one.
        counter = collections.Counter(real_names)
        merged_real_class = counter.most_common(1)[0][0]
        print("Warning: Not all real_class values match in CSV; using the most common value:", merged_real_class)

    # Build final class names list: synthetic names from each sub-model, plus the merged real class.
    final_class_names = synthetic_names + [merged_real_class]

    # Test the merged model with dummy input.
    dummy = torch.randn(2, 3, 512, 512).to(device)
    out = merged(dummy)
    print("Merged model output shape:", out.shape)  # e.g. [2, N+1]

    # Save both the state_dict and the metadata.
    torch.save({
        'state_dict': merged.state_dict(),
        'metadata': {
            'class_names': final_class_names
        }
    }, args.output_path)
    print(f"Saved merged model with metadata => {args.output_path}")

if __name__ == '__main__':
    main()
