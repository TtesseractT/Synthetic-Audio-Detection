#!/usr/bin/env python3
# Company: Uhmbrella Ltd 2025
# Author: Sabian Hibbs
# Date: 2025-01-01
# Version: 1.0
# License: MIT


import argparse
import os
import shutil


"""
This script checks for overlapping audio files between the 'train' and 'test'
subdirectories of a given base directory. Each of these subdirectories is
expected to contain class folders (e.g. 'class0', 'class1', etc.).

For each class folder, the script groups files by a “group key”. In this example,
the group key is defined as the part of the filename before the first underscore.
For example, from:
    1f6999ff03018e9a_add_white_noise_0.002498277_Segment_134.wav
the group key is "1f6999ff03018e9a".

For each group that exists in both train and test, the script checks the number
of files. If the counts differ, the side with fewer files is considered “wrong.”
In report mode (the default) it prints a summary of counts; if run with --fix,
it moves all files from the minority side to the majority side (removing duplicates if needed).

Usage:
    Report mode (no files moved):
      python check_fix_overlaps.py -i /path/to/base_dir

    Fix mode (move files from the smaller subfolder into the larger one):
      python check_fix_overlaps.py -i /path/to/base_dir --fix

The directory structure expected is:
    base_dir/
       train/
           class0/
               *.wav
           class1/
               *.wav
       test/
           class0/
               *.wav
           class1/
               *.wav
"""


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Check for overlapping audio files between train and test and optionally fix them."
    )
    parser.add_argument("-i", "--input_dir", required=True,
                        help="Base directory containing 'train' and 'test' subdirectories with class folders")
    parser.add_argument("--fix", action="store_true",
                        help="If provided, move files from the minority side into the majority side for each overlapping group")
    return parser.parse_args()

def extract_group_key(filename):
    """
    Extracts the group key from a filename.
    Example: "1f6999ff03018e9a_add_white_noise_0.002498277_Segment_134.wav"
             returns "1f6999ff03018e9a"
    If no underscore is found, the key is the filename without extension.
    """
    if "_" in filename:
        return filename.split("_")[0]
    return os.path.splitext(filename)[0]

def get_files_by_group(folder):
    """
    Returns a dictionary mapping group keys to a list of filenames in the given folder.
    Only .wav files are considered.
    """
    groups = {}
    if not os.path.isdir(folder):
        return groups
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".wav"):
            continue
        key = extract_group_key(fname)
        groups.setdefault(key, []).append(fname)
    return groups

def process_class(class_name, base_dir, do_fix=False):
    """
    For the given class (e.g. "class0"), look inside base_dir/train/class_name and base_dir/test/class_name.
    For each group key that appears in both, if the number of files differ, the side with fewer files is flagged.
    In fix mode, all files from the minority side are moved to the side with the larger count.
    
    Returns a summary dictionary for this class.
    """
    summary = {
        'class': class_name,
        'overlap_groups': {},  # each key maps to a dict: { 'train': count, 'test': count, 'moved': X }
        'total_wrong_train': 0,
        'total_wrong_test': 0
    }
    
    train_folder = os.path.join(base_dir, "train", class_name)
    test_folder = os.path.join(base_dir, "test", class_name)
    
    # If one of the class folders is missing, nothing to do.
    if not os.path.isdir(train_folder) or not os.path.isdir(test_folder):
        return summary

    train_groups = get_files_by_group(train_folder)
    test_groups = get_files_by_group(test_folder)
    
    # Only process group keys that appear in both train and test.
    common_keys = set(train_groups.keys()).intersection(set(test_groups.keys()))
    
    for key in common_keys:
        count_train = len(train_groups[key])
        count_test = len(test_groups[key])
        # If the counts are equal, we assume the overlap is as expected.
        if count_train == count_test:
            continue

        # Determine which folder has fewer files.
        if count_train > count_test:
            correct_folder = train_folder
            wrong_folder = test_folder
            wrong_count = count_test
            summary['total_wrong_test'] += wrong_count
            wrong_files = test_groups[key]
        else:
            correct_folder = test_folder
            wrong_folder = train_folder
            wrong_count = count_train
            summary['total_wrong_train'] += wrong_count
            wrong_files = train_groups[key]
        
        moved = 0
        if do_fix:
            # Move each file from the wrong folder to the correct folder.
            for fname in wrong_files:
                src = os.path.join(wrong_folder, fname)
                dst = os.path.join(correct_folder, fname)
                # If the file already exists in the correct folder, remove the source file.
                if os.path.exists(dst):
                    try:
                        os.remove(src)
                        moved += 1
                    except Exception:
                        pass
                else:
                    try:
                        shutil.move(src, dst)
                        moved += 1
                    except Exception:
                        pass
        summary['overlap_groups'][key] = {
            'train': count_train,
            'test': count_test,
            'moved': moved
        }
    return summary

def get_class_names(base_dir):
    """
    Returns a sorted list of class folder names found in either the train or test subdirectories.
    """
    class_names = set()
    for sub in ["train", "test"]:
        sub_dir = os.path.join(base_dir, sub)
        if os.path.isdir(sub_dir):
            for d in os.listdir(sub_dir):
                if os.path.isdir(os.path.join(sub_dir, d)):
                    class_names.add(d)
    return sorted(class_names)

def main():
    args = parse_arguments()
    base_dir = args.input_dir
    do_fix = args.fix

    class_names = get_class_names(base_dir)
    if not class_names:
        print("No class folders found in 'train' or 'test' subdirectories.")
        return

    overall_wrong_train = 0
    overall_wrong_test = 0
    report_lines = []

    for cls in class_names:
        summary = process_class(cls, base_dir, do_fix=do_fix)
        report_lines.append(f"Class '{cls}':")
        for key, data in summary['overlap_groups'].items():
            # Report the counts for each overlapping group.
            report_lines.append(f"  Group '{key}': train = {data['train']}, test = {data['test']}, " +
                                (f"moved = {data['moved']}" if do_fix else f"wrong = {min(data['train'], data['test'])}"))
        report_lines.append(f"  Total wrong in train: {summary['total_wrong_train']}")
        report_lines.append(f"  Total wrong in test: {summary['total_wrong_test']}\n")
        overall_wrong_train += summary['total_wrong_train']
        overall_wrong_test += summary['total_wrong_test']

    print("OVERLAP REPORT:")
    print("----------------")
    for line in report_lines:
        print(line)
    print("----------------")
    print(f"Overall wrong in train: {overall_wrong_train}")
    print(f"Overall wrong in test: {overall_wrong_test}")
    if do_fix:
        print("Fix mode enabled: Files from the smaller side have been moved into the larger side.")

if __name__ == "__main__":
    main()
