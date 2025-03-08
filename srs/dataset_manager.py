#!/usr/bin/env python3
"""
This script takes an input directory containing class folders (e.g. "class0", "class1"),
each with audio files (e.g. "1f6999ff03018e9a_add_white_noise_0.002498277_Segment_134.wav"),
and splits the files for each class into Train and Test sets based on the split ratio (-s).

The output directory will be structured as:
    output_dir/
        train/
            class0/   <-- contains train files for class0
            class1/   <-- contains train files for class1
        test/
            class0/   <-- contains test files for class0
            class1/   <-- contains test files for class1

For example, if -s 0.5, roughly 50% of the files in each class folder will be moved to train and 50% to test.
If -s 0.8, then 80% of the files go to train and the remaining 20% go to test.
"""

import argparse
import os
import shutil
import random
import concurrent.futures
import threading
from tqdm import tqdm

# Global lock to ensure thread-safe file moves and progress bar updates
move_lock = threading.Lock()
pbar = None  # Global progress bar

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Split audio files from class folders into Train and Test sets based on split ratio."
    )
    parser.add_argument("-i", "--input_dir", required=True,
                        help="Input directory containing class folders with audio files")
    parser.add_argument("-o", "--output_dir", required=True,
                        help="Output directory to store split audio files (with train/test subdirectories)")
    parser.add_argument("-s", "--split", type=float, default=0.5,
                        help="Split ratio for train (e.g., 0.5 means 50%% train, 50%% test; default: 0.5)")
    parser.add_argument("-t", "--threads", type=int, default=1,
                        help="Number of threads to use (default: 1)")
    return parser.parse_args()

def count_files(input_dir):
    """Count total number of .wav files in all class folders."""
    total = 0
    for folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(".wav"):
                total += 1
    return total

def safe_move(source, destination):
    """Thread-safe file move with progress bar update."""
    global pbar
    with move_lock:
        try:
            shutil.move(source, destination)
        except Exception:
            # Silently ignore errors
            pass
        pbar.update(1)

def process_class(class_folder, input_dir, output_dir, split_ratio):
    """
    For a given class folder, list all .wav files,
    randomly split them into train and test sets based on the split ratio,
    and move the files into the corresponding output subdirectories.
    """
    source_folder = os.path.join(input_dir, class_folder)
    if not os.path.isdir(source_folder):
        return

    # Get list of all .wav files in this class folder.
    files = [f for f in os.listdir(source_folder) if f.lower().endswith(".wav")]
    if not files:
        return

    # Determine number of train files (round to nearest integer)
    num_train = int(round(split_ratio * len(files)))
    # Randomly select files for train
    train_files = set(random.sample(files, num_train))
    # The remainder go to test
    test_files = set(files) - train_files

    # Create output subdirectories for this class
    train_output = os.path.join(output_dir, "train", class_folder)
    test_output = os.path.join(output_dir, "test", class_folder)
    os.makedirs(train_output, exist_ok=True)
    os.makedirs(test_output, exist_ok=True)

    # Move train files
    for f in train_files:
        src = os.path.join(source_folder, f)
        dst = os.path.join(train_output, f)
        safe_move(src, dst)

    # Move test files
    for f in test_files:
        src = os.path.join(source_folder, f)
        dst = os.path.join(test_output, f)
        safe_move(src, dst)

def main():
    global pbar
    args = parse_arguments()

    # Validate input directory exists
    if not os.path.exists(args.input_dir):
        exit(1)

    # Create output base directories for train and test sets
    for sub in ["train", "test"]:
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)

    # Count the total number of .wav files to move for progress bar
    total_files = count_files(args.input_dir)
    pbar = tqdm(total=total_files, desc="Moving files", unit="file", leave=True)

    # List class folders in the input directory
    class_folders = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
    if not class_folders:
        exit(1)

    # Process each class folder using specified number of threads
    if args.threads > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = [
                executor.submit(process_class, class_folder, args.input_dir, args.output_dir, args.split)
                for class_folder in class_folders
            ]
            concurrent.futures.wait(futures)
    else:
        for class_folder in class_folders:
            process_class(class_folder, args.input_dir, args.output_dir, args.split)

    pbar.close()

if __name__ == "__main__":
    main()
