import os
import pandas as pd
import numpy as np
from yt_dlp import YoutubeDL
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import logging

<<<<<<< HEAD
csv_path = "./datasets/shs100k/shs100k.csv"
output_dir = ""
=======
def parse_args():
    parser = argparse.ArgumentParser(description="Download audio files using YouTube IDs from metadata.")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="./bytecover/data/interim/shs100k.csv",
        help="Path to the metadata CSV file containing video IDs and metadata."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./bytecover/data/shs100k",
        help="Directory where the downloaded audio files will be saved."
    )
    parser.add_argument(
        "--train_ids_path",
        type=str,
        default="./bytecover/data/splits/train_ids.npy",
        help="Path to the NumPy file containing training IDs."
    )
    parser.add_argument(
        "--val_ids_path",
        type=str,
        default="./bytecover/data/splits/val_ids.npy",
        help="Path to the NumPy file containing validation IDs."
    )
    parser.add_argument(
        "--test_ids_path",
        type=str,
        default="./bytecover/data/splits/test_ids.npy",
        help="Path to the NumPy file containing test IDs."
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=16,
        help="Number of threads to use for downloading. Default is 16."
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=50,
        help="Number of files to download from the metadata. Default is 50."
    )
    return parser.parse_args()
>>>>>>> origin/frontend_improvements

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

<<<<<<< HEAD
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_path)

focused_train_ids = np.load("./datasets/shs100k/train_ids.npy", allow_pickle=True)
focused_val_ids = np.load("./datasets/shs100k/val_ids.npy", allow_pickle=True)
focused_test_ids = np.load("./datasets/shs100k/test_ids.npy", allow_pickle=True)


required_ids = set(focused_train_ids) | set(focused_val_ids) | set(focused_test_ids)

filtered_metadata = df[df["id"].isin(required_ids)]
print(f"Total files available: {len(filtered_metadata)}")

filtered_metadata = filtered_metadata.head(50)
print(f"Downloading first {len(filtered_metadata)} files.")


ydl_opts = {
    "format": "bestaudio/best",
    "postprocessors": [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "m4a",
            "preferredquality": "192",
        }
    ],
    "quiet": True,
    "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
}


def download_audio(row):
=======
def download_audio(row, ydl_opts, output_dir):
>>>>>>> origin/frontend_improvements
    """
    Downloads and renames the audio file based on metadata row.
    Ensures output is MP3 and named after the 'id' column.
    """
    video_id = row["Video ID"]
    file_id = row["id"]
    output_file = os.path.join(output_dir, f"{file_id}.mp3")

    # Check if the file already exists
    if os.path.exists(output_file):
        return f"File already exists: {output_file}"

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

<<<<<<< HEAD
=======
        # Rename downloaded file
>>>>>>> origin/frontend_improvements
        for file in os.listdir(output_dir):
            if file.startswith(video_id) and file.endswith(".mp3"):
                new_name = f"{file_id}.mp3"
                os.rename(
                    os.path.join(output_dir, file),
                    os.path.join(output_dir, new_name),
                )
                return f"Renamed {file} to {new_name}"
        return f"Download succeeded but no MP3 found for {video_id}"
    except Exception as e:
        return f"Failed to download {video_id}: {e}"

def main():
    args = parse_args()
    setup_logging()

<<<<<<< HEAD
num_threads = 16
results = []
=======
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.csv_path)
>>>>>>> origin/frontend_improvements

    focused_train_ids = np.load(args.train_ids_path, allow_pickle=True)
    focused_val_ids = np.load(args.val_ids_path, allow_pickle=True)
    focused_test_ids = np.load(args.test_ids_path, allow_pickle=True)

    required_ids = set(focused_train_ids) | set(focused_val_ids) | set(focused_test_ids)

    filtered_metadata = df[df["id"].isin(required_ids)]
    logging.info(f"Total files available: {len(filtered_metadata)}")

    filtered_metadata = filtered_metadata.head(args.sample_size)
    logging.info(f"Downloading first {len(filtered_metadata)} files.")

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "outtmpl": os.path.join(args.output_dir, "%(id)s.%(ext)s"),
    }

<<<<<<< HEAD
    for future in tqdm(
        as_completed(futures), total=len(futures), desc="Downloading Audio"
    ):
        try:
            results.append(future.result())
        except Exception as e:
            results.append(f"Error: {e}")
=======
    results = []
>>>>>>> origin/frontend_improvements

    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = {
            executor.submit(download_audio, row, ydl_opts, args.output_dir): row["Video ID"]
            for _, row in filtered_metadata.iterrows()
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading Audio"):
            try:
                results.append(future.result())
            except Exception as e:
                results.append(f"Error: {e}")

    for result in results:
        logging.info(result)

if __name__ == "__main__":
    main()
