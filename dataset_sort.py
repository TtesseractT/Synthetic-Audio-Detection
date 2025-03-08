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
         This includes audio files with the same name but different content.
         The name of each input directory audio file will have the following format:
             {original_file_name}_{augmentation_technique}_{random_value}.wav

         The original_file_name name cannot be duplicated between the 'Test' and 'Train' directories.

         This means that if there is 20 variations of one {original_file_name} in the input directory,
         these 20 audio files with the same {original_file_name} will be in the same directory
         (either 'Test' or 'Train'). The audio files will be randomly split between the 'Test' and 'Train' directories depending on the split ratio.

7. Move the audio files to the respective directories based on the class labels
"""

import argparse
import os
import shutil
import random
import concurrent.futures
import threading

# Global lock to ensure thread-safe file operations
move_lock = threading.Lock()

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Sort audio files into Train and Test directories based on class labels."
    )
    parser.add_argument("-i", "--input_dir", required=True, help="Input directory containing audio files")
    parser.add_argument("-o", "--output_dir", required=True, help="Output directory to store sorted audio files")
    parser.add_argument("-s", "--split", type=float, default=0.8, help="Split ratio for train and test data (default: 0.8)")
    parser.add_argument("-t", "--threads", type=int, default=1, help="Number of threads to use (default: 1)")

    return parser.parse_args()

def safe_move(source_file, target_file):
    """Moves a file in a thread-safe manner."""
    with move_lock:
        shutil.move(source_file, target_file)

def process_class(class_label, input_dir, output_dir, split_ratio):
    class_path = os.path.join(input_dir, class_label)
    if not os.path.isdir(class_path):
        return

    # Create class directories in both Train and Test outputs
    train_class_dir = os.path.join(output_dir, "Train", class_label)
    test_class_dir = os.path.join(output_dir, "Test", class_label)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    # Group audio files by original file name
    groups = {}
    for filename in os.listdir(class_path):
        if not filename.lower().endswith(".wav"):
            continue
        # Expecting format: {original_file_name}_{augmentation_technique}_{random_value}.wav
        parts = filename.split("_")
        if len(parts) < 3:
            print(f"Skipping file {filename} in {class_label}: filename format does not match.")
            continue
        original = parts[0]
        groups.setdefault(original, []).append(filename)

    # For each group, assign all files to either Train or Test based on split ratio
    for original, file_list in groups.items():
        target_dir = train_class_dir if random.random() < split_ratio else test_class_dir
        for file in file_list:
            source_file = os.path.join(class_path, file)
            target_file = os.path.join(target_dir, file)
            safe_move(source_file, target_file)

def main():
    args = parse_arguments()

    # Check if the input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        exit(1)

    # Create the output directory along with Train and Test subdirectories if they don't exist
    for sub in ["Train", "Test"]:
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)

    # Get list of class labels (subdirectories in the input directory)
    class_labels = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
    if not class_labels:
        print("No class label directories found in the input directory.")
        exit(1)

    # Process each class directory using the specified number of threads
    if args.threads > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = [
                executor.submit(process_class, class_label, args.input_dir, args.output_dir, args.split)
                for class_label in class_labels
            ]
            concurrent.futures.wait(futures)
    else:
        for class_label in class_labels:
            process_class(class_label, args.input_dir, args.output_dir, args.split)

    print("Audio files have been successfully sorted into Train and Test directories.")

if __name__ == "__main__":
    main()
