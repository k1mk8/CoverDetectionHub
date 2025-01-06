import wave
import random
import os
import argparse
from pydub import AudioSegment
from pydub.generators import WhiteNoise
import csv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def inject_audio(song_path, target_paths, percentage, output_dir, dir_percentage, random_seed):
    """
    Extract chunks of the song and insert them into target WAV files at random positions.
    Replace target segments with silence to make the injected song fragment the only audible part.

    :param song_path: Path to the main WAV file.
    :param target_paths: List of target WAV file paths.
    :param percentage: Percentage (0-100) of the song chunk to extract and insert into target files.
    :param output_dir: Directory to save the modified WAV files.
    :param dir_percentage: Percentage of files in each subfolder to inject.
    :param random_seed: Seed for random number generator.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    random.seed(random_seed)

    original_song_output_path = os.path.join(output_dir, os.path.basename(song_path))
    AudioSegment.from_wav(song_path).export(original_song_output_path, format="wav")
    logging.info(f"Original song saved to: {original_song_output_path}")


    song = AudioSegment.from_wav(song_path)

    # Prepare CSV file for logging
    csv_file_path = os.path.join(output_dir, "injection_list.csv")
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["File", "Injected"])

        for target_path in target_paths:
            # Load the target file
            target_audio = AudioSegment.from_wav(target_path)

            # Determine the chunk duration based on the percentage
            chunk_duration = len(song) * (percentage / 100)
            chunk_start = random.randint(0, len(song) - int(chunk_duration))
            chunk = song[chunk_start:chunk_start + int(chunk_duration)]

      
            inject = random.random() < (dir_percentage / 100)

            if inject:
                injection_points = [
                    random.randint(0, len(target_audio) - len(chunk))
                    for _ in range(random.randint(1, 5))
                ]

                modified_audio = target_audio
                for point in injection_points:
                    silence = AudioSegment.silent(duration=len(chunk))
                    modified_audio = modified_audio[:point] + silence + modified_audio[point + len(chunk):]
                    modified_audio = modified_audio.overlay(chunk, position=point)

                output_path = os.path.join(output_dir, f"{os.path.basename(target_path)}_injection.wav")
                modified_audio.export(output_path, format="wav")
                logging.info(f"Injected and saved: {output_path}")

                csv_writer.writerow([target_path, "Yes"])
            else:
                # Copy the original file unchanged
                unchanged_output_path = os.path.join(output_dir, os.path.basename(target_path))
                target_audio.export(unchanged_output_path, format="wav")
                logging.info(f"Unchanged file saved to: {unchanged_output_path}")
                csv_writer.writerow([target_path, "No"])

def create_white_noise_with_injections(song_path, duration_ms, percentage, output_dir, csv_writer):
    """
    Create a white noise file and insert chunks of the song into it.
    Replace segments of the white noise with silence so only the song chunk is audible.

    :param song_path: Path to the main WAV file.
    :param duration_ms: Duration of the white noise in milliseconds.
    :param percentage: Percentage (0-100) of the song chunk to extract and insert.
    :param output_dir: Directory to save the modified white noise files.
    :param csv_writer: CSV writer object to log the injections.
    """
    white_noise = WhiteNoise().to_audio_segment(duration=duration_ms)

    # Load the main song
    song = AudioSegment.from_wav(song_path)

    # Determine the chunk duration based on the percentage
    chunk_duration = len(song) * (percentage / 100)
    chunk_start = random.randint(0, len(song) - int(chunk_duration))
    chunk = song[chunk_start:chunk_start + int(chunk_duration)]

    # Create 8 injections into the white noise
    for i in range(3):
        injection_points = [
            random.randint(0, len(white_noise) - len(chunk))
            for _ in range(random.randint(1, 5))  # Number of insertions per version
        ]

        modified_noise = white_noise
        for point in injection_points:
            silence = AudioSegment.silent(duration=len(chunk))
            modified_noise = modified_noise[:point] + silence + modified_noise[point + len(chunk):]
            modified_noise = modified_noise.overlay(chunk, position=point)

        output_path = os.path.join(output_dir, f"white_noise_injection_{i + 1}.wav")
        modified_noise.export(output_path, format="wav")
        logging.info(f"Injected and saved: {output_path}")

        csv_writer.writerow([output_path, "Yes"])

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insert extracted audio chunks into WAV files.")
    parser.add_argument("--input_folder", required=True, help="Folder containing target WAV files.")
    parser.add_argument("--song_file", required=True, help="Path to the main song file.")
    parser.add_argument("--output_folder", required=True, help="Directory to save the output files.")
    parser.add_argument("--percentage", type=float, required=True, help="Percentage of the song to extract and insert into each target file.")
    parser.add_argument("--dir_percentage", type=float, required=True, help="Percentage of files in each subfolder to inject.")
    parser.add_argument("--random_seed", type=int, default=42, help="Seed for random number generator.")

    args = parser.parse_args()

    # List all WAV files in the input folder and subfolders
    target_files = []
    for root, _, files in os.walk(args.input_folder):
        for f in files:
            if f.endswith(".wav"):
                target_files.append(os.path.join(root, f))

    # Prepare CSV file for logging
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    csv_file_path = os.path.join(args.output_folder, "injection_list.csv")
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["File", "Injected"])

        # Insert extracted audio into each file and organize into output folder
        inject_audio(
            args.song_file,
            target_files,
            percentage=args.percentage,
            output_dir=args.output_folder,
            dir_percentage=args.dir_percentage,
            random_seed=args.random_seed
        )

        # Create a 3-minute white noise file and insert samples into it
        create_white_noise_with_injections(
            args.song_file, duration_ms=3 * 60 * 1000, percentage=args.percentage, output_dir=args.output_folder, csv_writer=csv_writer
        )
