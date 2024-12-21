import os
import pandas as pd
import json
import wave
import argparse

def get_wav_duration(wav_file_path):
    try:
        with wave.open(wav_file_path, 'rb') as wav_file:
            num_frames = wav_file.getnframes()
            frame_rate = wav_file.getframerate()
            duration = num_frames / float(frame_rate)
            return duration
    except Exception as e:
        print(f"Error processing file {wav_file_path}: {e}")
        return None

def main(data_folder, csv_file):
    df = pd.read_csv(csv_file)

    json_data = []

    for index, row in df.iterrows():
        clique = row['clique']
        version = row['version']
        title = row['title']
        performer = row['performer']
        video_id = row['Video ID']
        
        wav_file_path = os.path.join(data_folder, f"{video_id}.wav")
        
        if os.path.exists(wav_file_path):
            duration = get_wav_duration(wav_file_path)
            
            if duration is not None:
                json_entry = {
                    "utt": f"cover{clique:02d}_{index:08d}_{version}_{index % 2}",
                    "wav": f"data/covers80/wav_16k/{video_id}.wav",
                    "dur_s": duration,
                    "song": title,
                    "version": f"{performer}+{title}+{version}-{title}.mp3"
                }
                json_data.append(json_entry)

    output_file = 'output_data.json'
    with open(output_file, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

    print(f"JSON file has been created: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio files and CSV to generate JSON data")
    parser.add_argument('data_folder', type=str, help="Path to the directory containing WAV files")
    parser.add_argument('csv_file', type=str, help="Path to the CSV file")

    args = parser.parse_args()

    main(args.data_folder, args.csv_file)
