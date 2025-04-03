#!/usr/bin/env python3
# Company:  Uhmbrella Ltd 2025
# Author:   Sabian Hibbs
# Date:     2025-01-01
# Version:  1.0
# License:  MIT


import argparse
import librosa
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import multiprocessing
import soundfile as sf
from scipy.signal import lfilter

'''
This script will use librosa library and others to augment the audio files in the dataset.

Arguments: [-i, -o]
1 input file = 11 augmented files (1 original + 10 augmented)
1 output folder location for all augmented files

Augmentation techniques:
1. Time stretching [min: 0.5, max: 1.5]
    1.1. Speed up random amount 
    1.2. Slow down random amount         
2. Pitch shifting [min: -2, max: 2]
    2.1. Increase pitch random amount
    2.2. Decrease pitch random amount
3. Dynamic range compression [min: 0.5, max: 1.5]
    3.1. Increase Compression random amount
    3.2. Decrease Compression random amount
4. Adding white noise [min volume: 0.001, max volume: 0.6]
    4.1 Add white noise random amount
5. Adding phase shift [min: -180, max: 180]
    5.1 Add phase shift random amount
6. Adding random filtering [min: 0, max: 4]
    6.1 Add random filtering (Band pass, Band stop, High pass, Low pass) # but not all at once.
7. Adding random shift in the time domain [min: -0.5, max: 0.5]
    7.1 Add random shift in the time domain (mixup of time stretching and pitch shifting)
8. Original file 

The name of each file will have the following format:
{original_file_name}_{augmentation_technique}_{random_value}.wav # STEREO, 44.1kHz, 16-bit PCM

Script will use ALL available cores on system to run the augmentations in parallel.
    Output of all augmented files saved to a CSV file for future reference.

Progress of the augmentation will be displayed in the console via TQDM progress bar - nothing else. 
'''

def augment_speed_up(y, min_rate=1.0, max_rate=1.5):
    rate = np.random.uniform(min_rate, max_rate)
    y_stretched = librosa.effects.time_stretch(y, rate=rate)
    return y_stretched, rate


def augment_slow_down(y, min_rate=0.5, max_rate=1.0):
    rate = np.random.uniform(min_rate, max_rate)
    y_stretched = librosa.effects.time_stretch(y, rate=rate)
    return y_stretched, rate


def augment_pitch_up(y, sr, min_steps=0, max_steps=2):
    n_steps = np.random.uniform(min_steps, max_steps)
    y_shifted = librosa.effects.pitch_shift(y, n_steps=n_steps, sr=sr)
    return y_shifted, n_steps


def augment_pitch_down(y, sr, min_steps=-2, max_steps=0):
    n_steps = np.random.uniform(min_steps, max_steps)
    y_shifted = librosa.effects.pitch_shift(y, n_steps=n_steps, sr=sr)
    return y_shifted, n_steps


def augment_dynamic_range_compression(y, min_amount=0.01, max_amount=0.5):
    amount = np.random.uniform(min_amount, max_amount)
    y_compressed = np.sign(y) * (np.abs(y) ** amount)
    return y_compressed, amount


def augment_add_white_noise(y, min_vol=0.001, max_vol=0.05):
    rms = np.sqrt(np.mean(y ** 2))
    noise_amp = np.random.uniform(min_vol, max_vol) * rms
    noise = noise_amp * np.random.normal(size=y.shape[0])
    y_noisy = y + noise
    return y_noisy, noise_amp


def augment_tremolo(y, sr, min_rate=3.0, max_rate=6.0, min_depth=0.2, max_depth=0.5):
    """
    Apply a tremolo effect by modulating the amplitude of the signal with a low-frequency oscillator (LFO).
    """
    lfo_rate = np.random.uniform(min_rate, max_rate)  # LFO frequency in Hz
    depth = np.random.uniform(min_depth, max_depth)   # Depth of modulation
    t = np.linspace(0, len(y) / sr, num=len(y))
    lfo = (1 - depth) + depth * np.sin(2 * np.pi * lfo_rate * t)
    y_tremolo = y * lfo
    param = {'lfo_rate': lfo_rate, 'depth': depth}
    return y_tremolo, param


def augment_phaser(y, sr, min_rate=0.1, max_rate=1.0, min_depth=0.5, max_depth=0.9):
    """
    Apply a phaser effect to the audio signal using a simplified implementation.
    """
    depth = np.random.uniform(min_depth, max_depth)
    rate = np.random.uniform(min_rate, max_rate)
    t = np.arange(len(y)) / sr
    lfo = depth * np.sin(2 * np.pi * rate * t)

    # Use multiple all-pass filters at different center frequencies
    y_phased = y.copy()
    for f0 in [500, 1500, 2500]:  # Center frequencies in Hz
        omega = 2 * np.pi * f0 / sr
        alpha = np.sin(omega) / 2
        b = [alpha, 0, -alpha]
        a = [1 + alpha, -2 * np.cos(omega), 1 - alpha]
        y_filtered = lfilter(b, a, y_phased)
        y_phased += lfo * y_filtered  # Modulate with LFO and add to output

    param = {'rate': rate, 'depth': depth}
    return y_phased, param


def augment_time_shift(y, sr, min_shift=-0.5, max_shift=0.5):
    shift = np.random.uniform(min_shift, max_shift)
    shift_samples = int(shift * sr)
    y_shifted = np.roll(y, shift_samples)
    if shift_samples > 0:
        y_shifted[:shift_samples] = 0
    else:
        y_shifted[shift_samples:] = 0
    return y_shifted, shift


def augment_time_pitch_shift(y, sr):
    rate = np.random.uniform(0.8, 1.2)
    n_steps = np.random.uniform(-1, 1)
    y_stretched = librosa.effects.time_stretch(y, rate=rate)
    y_shifted = librosa.effects.pitch_shift(y_stretched, n_steps=n_steps, sr=sr)
    return y_shifted, (rate, n_steps)


def process_augmentation(task):
    input_file, output_folder, augmentation_name = task
    try:
        y, sr = librosa.load(input_file, sr=44100, mono=True)

        if augmentation_name == 'original':
            y_aug = y
            param = None
        elif augmentation_name == 'speed_up':
            y_aug, rate = augment_speed_up(y)
            param = rate
        elif augmentation_name == 'slow_down':
            y_aug, rate = augment_slow_down(y)
            param = rate
        elif augmentation_name == 'pitch_up':
            y_aug, n_steps = augment_pitch_up(y, sr)
            param = n_steps
        elif augmentation_name == 'pitch_down':
            y_aug, n_steps = augment_pitch_down(y, sr)
            param = n_steps
        elif augmentation_name == 'dynamic_range_compression':
            y_aug, amount = augment_dynamic_range_compression(y)
            param = amount
        elif augmentation_name == 'add_white_noise':
            y_aug, noise_amp = augment_add_white_noise(y)
            param = noise_amp
        elif augmentation_name == 'tremolo':
            y_aug, tremolo_params = augment_tremolo(y, sr)
            param = tremolo_params
        elif augmentation_name == 'phaser':
            y_aug, phaser_params = augment_phaser(y, sr)
            param = phaser_params
        elif augmentation_name == 'time_shift':
            y_aug, shift = augment_time_shift(y, sr)
            param = shift
        elif augmentation_name == 'time_pitch_shift':
            y_aug, (rate, n_steps) = augment_time_pitch_shift(y, sr)
            param = f'rate_{rate}_steps_{n_steps}'
        else:
            return None

        # Ensure the audio is within the valid range [-1, 1]
        y_aug = np.clip(y_aug, -1.0, 1.0)

        y_aug_stereo = np.stack((y_aug, y_aug), axis=-1)

        base_name = os.path.splitext(os.path.basename(input_file))[0]
        if param is not None:
            param_str = str(param).replace(' ', '_').replace(',', '_').replace(':', '_').replace('{', '').replace('}', '')
            output_file = f"{base_name}_{augmentation_name}_{param_str}.wav"
        else:
            output_file = f"{base_name}_{augmentation_name}.wav"
        output_path = os.path.join(output_folder, output_file)

        sf.write(output_path, y_aug_stereo, sr, subtype='PCM_16')

        return {
            'input_file': input_file,
            'output_file': output_file,
            'augmentation': augmentation_name,
            'param': param,
        }
    except Exception as e:
        print(f"Error processing {input_file} with {augmentation_name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Audio Augmentation Script')
    parser.add_argument('-i', '--input', required=True, help='Input file or folder')
    parser.add_argument('-o', '--output', required=True, help='Output folder')
    parser.add_argument('-c', '--csv', required=False, help='CSV output file path')
    parser.add_argument('-p', '--pool-size', type=int, default=multiprocessing.cpu_count(),
                      help='Number of processes in the pool (default: number of CPU cores)')
    args = parser.parse_args()

    input_path = args.input
    output_folder = args.output
    csv_output = args.csv
    pool_size = args.pool_size

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if os.path.isfile(input_path):
        if input_path.lower().endswith(('.wav', '.mp3')):
            input_files = [input_path]
        else:
            print(f'Input file must be a .wav or .mp3 file: {input_path}')
            exit(1)
    elif os.path.isdir(input_path):
        input_files = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.lower().endswith(('.wav', '.mp3'))
        ]
        if not input_files:
            print(f'No .wav or .mp3 files found in directory: {input_path}')
            exit(1)
    else:
        print(f'Input path is not a valid file or directory: {input_path}')
        exit(1)
    
    print(f'Found {len(input_files)} audio files to process.')

    augmentations = [
        'original',
        'speed_up',
        'slow_down',
        'pitch_up',
        'pitch_down',
        'dynamic_range_compression',
        'add_white_noise',
        'tremolo',
        'phaser',
        'time_shift',
        'time_pitch_shift',
    ]

    tasks = []
    for input_file in input_files:
        for augmentation in augmentations:
            tasks.append((input_file, output_folder, augmentation))

    pool = multiprocessing.Pool(processes=pool_size)
    results = []
    for res in tqdm(pool.imap_unordered(process_augmentation, tasks), total=len(tasks)):
        if res is not None:
            results.append(res)
    pool.close()
    pool.join()

    df = pd.DataFrame(results)

    if not csv_output:
        csv_output = os.path.join(output_folder, 'augmentation_results.csv')
    df.to_csv(csv_output, index=False)


if __name__ == '__main__':
    main()
