#!/usr/bin/env python3
"""
This script will take the input directory containing audio files
and sort them into the output directory based on the class labels.

Args:
-i (--input_dir): Input directory containing audio files
-o (--output_dir): Output directory to store sorted audio files
-s (--split): Split ratio for train and test data (default: 0.8)
-h (--help): Display help message
-t (--threads): Number of threads to use (default: 1)

Logic:
1. Parse the input arguments
2. Check if the input directory exists
3. Create the output directory if it does not exist
4. Get the list of class labels from the input directory
5. Split the class labels into train and test sets
6. Create the train and test directories in the output directory
   Note: No audio files are duplicated between the 'Test' and 'Train' directories.
7. Move the audio files to the respective directories based on the class labels

Input Data (audio files) Directory Structure:
    input_dir
    ├── class1  # class label 1
    │   ├── audio1.wav
    │   ├── audio2.wav
    │   └── ...
    └── class2  # class label 2
        ├── audio1.wav  # audio file 1
        ├── audio2.wav  # audio file 2
        └── ...

Output Data (audio files) Directory Structure:
    output_dir
        ├── Test
        │   ├── class1  # class label 1
        │   │   └── ... # audio files
        │   └── class2  # class label 2
        │       └── ... # audio files
        └── Train
            ├── class1  # class label 1
            │   └── ... # audio files
            └── class2  # class label 2
                └── ... # audio files

"""

import os
import shutil
import argparse
import random
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Set


def parse_arguments():
    parser = argparse.ArgumentParser(description='Sort audio files into train and test directories.')
    parser.add_argument('-i', '--input_dir', required=True, help='Input directory containing audio files')
    parser.add_argument('-o', '--output_dir', required=True, help='Output directory to store sorted audio files')
    parser.add_argument('-s', '--split', type=float, default=0.8, help='Split ratio for train and test data (default: 0.8)')
    parser.add_argument('-t', '--threads', type=int, default=1, help='Number of threads to use (default: 1)')
    return parser.parse_args()


def extract_original_filename(filename: str) -> str:
    """Extract the original filename without augmentation info and extension."""
    # Assumes filename format: original_name_augmentation_random.wav
    # If the file doesn't follow this pattern, return the filename without extension
    if '_' in filename:
        return filename.split('_')[0]
    return os.path.splitext(filename)[0]


def get_class_files(input_dir: str) -> Dict[str, List[str]]:
    """Get files organized by class."""
    class_files = {}
    
    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if os.path.isdir(class_path):
            files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
            class_files[class_name] = files
    
    return class_files


def split_files_by_original_name(class_files: Dict[str, List[str]], split_ratio: float) -> Dict[str, Dict[str, List[str]]]:
    """Split files into train and test sets based on original filenames."""
    result = {"train": {}, "test": {}}
    
    for class_name, files in class_files.items():
        # Group files by original name
        original_name_groups = {}
        for file in files:
            original_name = extract_original_filename(file)
            if original_name not in original_name_groups:
                original_name_groups[original_name] = []
            original_name_groups[original_name].append(file)
        
        # Get unique original names and shuffle
        original_names = list(original_name_groups.keys())
        random.shuffle(original_names)
        
        # Split based on original names
        split_index = int(len(original_names) * split_ratio)
        train_names = original_names[:split_index]
        test_names = original_names[split_index:]
        
        # Collect files for train and test
        train_files = []
        test_files = []
        
        for name in train_names:
            train_files.extend(original_name_groups[name])
        
        for name in test_names:
            test_files.extend(original_name_groups[name])
        
        result["train"][class_name] = train_files
        result["test"][class_name] = test_files
    
    return result


def create_directory_structure(output_dir: str, class_names: List[str]):
    """Create the directory structure for output."""
    os.makedirs(output_dir, exist_ok=True)
    
    for split in ["Train", "Test"]:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        for class_name in class_names:
            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)


def copy_file(input_dir: str, output_dir: str, class_name: str, filename: str, split: str):
    """Copy a file from input directory to output directory."""
    src = os.path.join(input_dir, class_name, filename)
    dst = os.path.join(output_dir, split, class_name, filename)
    shutil.copy2(src, dst)


def main():
    args = parse_arguments()
    
    # Check if input directory exists
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        return
    
    # Get class names and files
    class_files = get_class_files(args.input_dir)
    class_names = list(class_files.keys())
    
    # Create output directory structure
    create_directory_structure(args.output_dir, class_names)
    
    # Split files into train and test sets
    split_files = split_files_by_original_name(class_files, args.split)
    
    # Copy files to output directories using multiple threads
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        # Copy train files
        for class_name, files in split_files["train"].items():
            for file in files:
                executor.submit(copy_file, args.input_dir, args.output_dir, class_name, file, "Train")
        
        # Copy test files
        for class_name, files in split_files["test"].items():
            for file in files:
                executor.submit(copy_file, args.input_dir, args.output_dir, class_name, file, "Test")
    
    print(f"Finished sorting files. Train/Test split: {args.split}/{1-args.split}")
    
    # Print statistics
    total_files = sum(len(files) for files in class_files.values())
    train_files = sum(len(files) for files in split_files["train"].values())
    test_files = sum(len(files) for files in split_files["test"].values())
    
    print(f"Total files: {total_files}")
    print(f"Train files: {train_files} ({train_files/total_files:.2%})")
    print(f"Test files: {test_files} ({test_files/total_files:.2%})")


if __name__ == "__main__":
    main()