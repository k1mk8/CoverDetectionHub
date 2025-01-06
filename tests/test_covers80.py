import os
import pytest
import random
import yaml

from evaluation.covers80_eval import gather_covers80_dataset_files

with open("configs/paths.yaml", "r") as f:
    config = yaml.safe_load(f)

COVERS80_DATASET_PATH = config["covers80_data_dir"]
COVERS80BUT10_DATASET_PATH = config["covers80_data_dir"]


def test_gather_covers80_dataset_files():
    """
    Test that gather_covers80_dataset_files returns a non-empty list of (audio_file_path, label).
    (This assumes the covers80 dataset is properly populated.)
    """
    results = gather_covers80_dataset_files(COVERS80_DATASET_PATH)
    assert len(results) > 0, "Expected at least one (audio_path, label) tuple from covers80 dataset."


def test_covers80_random_subfolders_have_two_mp3s():
    """
    Randomly pick a few subfolders in covers80 dataset directory and check
    if each contains at least 2 .mp3 files.
    """
    subfolders = [
        f for f in os.listdir(COVERS80_DATASET_PATH)
        if os.path.isdir(os.path.join(COVERS80_DATASET_PATH, f))
    ]

    if not subfolders:
        pytest.skip(f"No subfolders found in {COVERS80_DATASET_PATH}; cannot test .mp3 files.")

    num_to_check = min(3, len(subfolders))
    random_folders = random.sample(subfolders, k=num_to_check)

    for folder in random_folders:
        folder_path = os.path.join(COVERS80_DATASET_PATH, folder)
        mp3_files = [f for f in os.listdir(folder_path) if f.endswith(".mp3")]
        assert len(mp3_files) >= 2, (
            f"Folder '{folder}' should have at least 2 .mp3 files, "
            f"but found {len(mp3_files)}"
        )


def test_gather_covers80but10_dataset_files():
    """
    Test that gather_covers80_dataset_files returns a non-empty list of (audio_file_path, label).
    (This assumes the covers80 dataset is properly populated.)
    """
    results = gather_covers80_dataset_files(COVERS80BUT10_DATASET_PATH)
    assert len(results) > 0, "Expected at least one (audio_path, label) tuple from covers80 dataset."


def test_covers80but10_random_subfolders_have_two_mp3s():
    """
    Randomly pick a few subfolders in covers80 dataset directory and check
    if each contains at least 2 .mp3 files.
    """
    subfolders = [
        f for f in os.listdir(COVERS80BUT10_DATASET_PATH)
        if os.path.isdir(os.path.join(COVERS80BUT10_DATASET_PATH, f))
    ]

    if not subfolders:
        pytest.skip(f"No subfolders found in {COVERS80BUT10_DATASET_PATH}; cannot test .mp3 files.")

    num_to_check = min(3, len(subfolders))
    random_folders = random.sample(subfolders, k=num_to_check)

    for folder in random_folders:
        folder_path = os.path.join(COVERS80BUT10_DATASET_PATH, folder)
        mp3_files = [f for f in os.listdir(folder_path) if f.endswith(".mp3")]
        assert len(mp3_files) >= 2, (
            f"Folder '{folder}' should have at least 2 .mp3 files, "
            f"but found {len(mp3_files)}"
        )
