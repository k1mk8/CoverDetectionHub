import os
import pytest
import yaml

with open("configs/paths.yaml", "r") as f:
    config = yaml.safe_load(f)

@pytest.mark.parametrize("key", list(config.keys()))
def test_paths_exist(key):
    """
    Test that each path in configs/paths.yaml exists.
    If the key includes "dir", we expect a directory;
    otherwise, we expect a file.
    """
    path_value = config[key]

    # Basic heuristic to guess whether it's a directory or a file
    if "dir" in key or "checkpoint_dir" in key:
        assert os.path.isdir(path_value), (
            f"Expected '{key}' to be a directory, but '{path_value}' is missing or not a directory."
        )
    else:
        # By default, assume it's a file
        assert os.path.isfile(path_value), (
            f"Expected '{key}' to be a file, but '{path_value}' is missing or not a file."
        )
