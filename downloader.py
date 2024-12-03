import os
import pandas as pd
import numpy as np
from yt_dlp import YoutubeDL
from tqdm.notebook import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

csv_path = "./data/interim/shs100k.csv" 
output_dir = "./data/shs100k" 


os.makedirs(output_dir, exist_ok=True)


df = pd.read_csv(csv_path)


focused_train_ids = np.load("./data/splits/train_ids.npy", allow_pickle=True)
focused_val_ids = np.load("./data/splits/val_ids.npy", allow_pickle=True)
focused_test_ids = np.load("./data/splits/test_ids.npy", allow_pickle=True)


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
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }
    ],
    "quiet": True, 
    "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"), 
}


def download_audio(row):
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



num_threads = 16
results = []

with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = {
        executor.submit(download_audio, row): row["Video ID"]
        for _, row in filtered_metadata.iterrows()
    }

    for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading Audio"):
        try:
            results.append(future.result())
        except Exception as e:
            results.append(f"Error: {e}")

for result in results:
    print(result)
