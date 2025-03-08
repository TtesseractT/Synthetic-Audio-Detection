import os
import subprocess
import argparse
from multiprocessing import Pool
from tqdm import tqdm

def convert_audio_file(input_output_pair):
    """
    Converts an audio file to a specified format using ffmpeg.

    Args:
        input_output_pair (tuple): A tuple containing the input file path and the output file path.

    Returns:
        str: The path of the converted audio file.
    """
    input_file, output_file = input_output_pair
    command = [
        'ffmpeg',
        '-i', input_file,
        '-ar', '32000',  # Sample rate: 32000 Hz
        '-ac', '1',      # Mono channel
        '-sample_fmt', 's16',  # Bit depth: 16
        '-f', 'wav', output_file  # Output format: WAV
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_file


def main():
    parser = argparse.ArgumentParser(description='Convert audio files within a directory to specified format using ffmpeg and multiprocessing.')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to the input directory containing audio files.')
    parser.add_argument('-o', '--output', type=str, required=True, help='Path to the output directory where converted audio files will be saved.')
    args = parser.parse_args()

    input_folder = args.input
    output_folder = args.output

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    audio_files = [file for file in os.listdir(input_folder) if file.endswith(('.mp3', '.wav', '.ogg', '.flac', '.aac', '.wma', '.opus'))]
    input_output_pairs = [(os.path.join(input_folder, audio_file), os.path.join(output_folder, os.path.splitext(audio_file)[0] + '.wav')) for audio_file in audio_files]

    with Pool() as pool, tqdm(total=len(input_output_pairs), desc='Converting audio files') as pbar:
        for _ in pool.imap_unordered(convert_audio_file, input_output_pairs):
            pbar.update(1)

if __name__ == "__main__":
    main()
