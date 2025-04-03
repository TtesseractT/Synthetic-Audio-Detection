#!/usr/bin/env python3
# Company:  Uhmbrella Ltd 2025
# Author:   Sabian Hibbs
# Date:     2025-01-01
# Version:  1.0
# License:  MIT


import os
import hashlib
import argparse

AUDIO_EXTENSIONS = ('.mp3', '.wav', '.ogg', '.flac', '.aac', '.wma', '.opus')

def hash_file(file_path):
    """Generate a hash for a file and return the first 6 characters."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as afile:
        buf = afile.read()
        hasher.update(buf)
    return hasher.hexdigest()[:16]

def rename_files_in_directory(input_dir, recursive=False):
    if recursive:
        for root, _, files in os.walk(input_dir):
            for filename in files:
                if filename.lower().endswith(AUDIO_EXTENSIONS):
                    file_path = os.path.join(root, filename)
                    file_hash = hash_file(file_path)
                    extension = os.path.splitext(filename)[1]
                    new_filename = f"{file_hash}{extension}"
                    new_file_path = os.path.join(root, new_filename)
                    os.rename(file_path, new_file_path)
                    print(f"Renamed {filename} to {new_filename}")
    else:
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(AUDIO_EXTENSIONS):
                file_path = os.path.join(input_dir, filename)
                file_hash = hash_file(file_path)
                extension = os.path.splitext(filename)[1]
                new_filename = f"{file_hash}{extension}"
                new_file_path = os.path.join(input_dir, new_filename)
                os.rename(file_path, new_file_path)
                print(f"Renamed {filename} to {new_filename}")

def main():
    parser = argparse.ArgumentParser(description='Rename all audio files in a directory to the first 6 characters of their hash values.')
    parser.add_argument('-i', '--input_dir', type=str, required=True, help='The directory containing audio files to be renamed.')
    parser.add_argument('-r', '--recursive', action='store_true', help='Recursively process subdirectories')
    
    args = parser.parse_args()
    
    rename_files_in_directory(args.input_dir, args.recursive)

if __name__ == "__main__":
    main()
