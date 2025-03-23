#!/usr/bin/env python3
# Company: Uhmbrella Ltd 2025
# Author: Sabian Hibbs
# Date: 2025-01-01
# Version: 1.0
# License: MIT


import os
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def split_audio(input_filepath, output_dir):
    target_sample_rate = 32000
    segment_duration = 4  # seconds
    
    file_base, _ = os.path.splitext(os.path.basename(input_filepath))
    os.makedirs(output_dir, exist_ok=True)
    
    # Build the segment output pattern.
    # This will create files like: {file_base}_Segment_000.wav, {file_base}_Segment_001.wav, etc.
    segment_pattern = os.path.join(output_dir, f"{file_base}_Segment_%03d.wav")
    
    ffmpeg_cmd = [
        'ffmpeg', '-loglevel', 'error', '-i', input_filepath,
        '-ac', '1',                 # Convert to mono
        '-ar', str(target_sample_rate),  # Resample
        '-filter:a', 'pan=mono|c0=0.5*c0+0.5*c1',  # Mix channels to mono
        '-f', 'segment',
        '-segment_time', str(segment_duration),
        '-reset_timestamps', '1',
        segment_pattern
    ]
    
    subprocess.run(ffmpeg_cmd, check=True)

def process_files_in_directory(input_path, output_dir):
    if os.path.isdir(input_path):
        audio_files = [
            os.path.join(input_path, f) for f in os.listdir(input_path)
            if f.lower().endswith(('.wav', '.mp3', '.flac', '.aac', '.ogg'))
        ]
    else:
        audio_files = [input_path]
    
    total_files = len(audio_files)
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(split_audio, audio_file, output_dir): audio_file for audio_file in audio_files}
        with tqdm(total=total_files, desc="Processing audio files") as pbar:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    # Optionally log error or handle it differently.
                    pass
                pbar.update(1)

def main():
    parser = argparse.ArgumentParser(description='Split audio files into 4-second mono segments.')
    parser.add_argument('-i', '--input', required=True, help='Input directory or file path containing audio files.')
    parser.add_argument('-o', '--output', required=True, help='Output directory for segmented audio files.')
    
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    process_files_in_directory(args.input, args.output)

if __name__ == '__main__':
    main()
