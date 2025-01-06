'''
Audio files converter for CoverHunter (https://github.com/Liu-Feng-deeplearning/CoverHunter) model.
Our scraped datased is in m4a format with sr = 22.5kHz, so we needed to convert the files into wav format
and resample it to 16kHz up to the model requirements.
'''
import os
import ffmpeg
import librosa
import soundfile as sf
import argparse

def convert_and_resample(input_dir, output_dir, target_sr=16000, limit=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    processed_files = 0

    m4a_files = [f for f in os.listdir(input_dir) if f.endswith('.m4a')]

    if limit is not None:
        m4a_files = m4a_files[:limit]

    for file_name in m4a_files:
        input_path = os.path.join(input_dir, file_name)
        temp_wav_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + '_temp.wav')
        output_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + '.wav')

        try:
            print(f"Converting {file_name} to temporary WAV...")
            ffmpeg.input(input_path).output(temp_wav_path, format='wav').run(quiet=True, overwrite_output=True)
            print(f"Converted {file_name} to temporary WAV.")

            y, sr = librosa.load(temp_wav_path, sr=None)
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sf.write(output_path, y_resampled, target_sr)
            print(f"Saved resampled file: {output_path}")

            os.remove(temp_wav_path)
            print(f"Removed temporary file: {temp_wav_path}")

            processed_files += 1

        except Exception as e:
            print(f"Failed to process {file_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert .m4a files to .wav and resample them.")
    parser.add_argument('input_dir', type=str, help="Directory containing .m4a files")
    parser.add_argument('output_dir', type=str, help="Directory to save the .wav files")
    parser.add_argument('--target_sr', type=int, default=16000, help="Target sampling rate (default is 16000 Hz)")
    parser.add_argument('--limit', type=int, default=None, help="Limit the number of files to process (default is no limit)")

    args = parser.parse_args()

    convert_and_resample(args.input_dir, args.output_dir, target_sr=args.target_sr, limit=args.limit)

if __name__ == "__main__":
    main()
