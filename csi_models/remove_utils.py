import os
import numpy as np

import essentia.standard as estd
import torch

from feature_extraction.audio_preprocessing import preprocess_audio

# Configuration


REMOVE_CHECKPOINT_DIR = ""
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def crema(audio_file, fs=44100, hop_length=512):
    """
    Compute "convolutional and recurrent estimators for music analysis" (CREMA)
    and resample so that it's reported in hop_length intervals
    NOTE: This code is a bit finnecky, and is recommended for Python 3.5.
    Check `wrapper_cream_feature` for the actual implementation.

    Returns
    -------
    crema: ndarray(n_frames, 12)
        The crema coefficients at each frame
    """
    from crema.models.chord import ChordModel
    from scipy import interpolate

    audio_vector = estd.MonoLoader(filename=audio_file, sampleRate=fs)()

    model = ChordModel()
    data = model.outputs(y=audio_vector, sr=fs)
    fac = (float(fs) / 44100.0) * 4096.0 / hop_length
    times_orig = fac * np.arange(len(data["chord_bass"]))
    nwins = int(np.floor(float(audio_vector.size) / hop_length))
    times_new = np.arange(nwins)
    interp = interpolate.interp1d(
        times_orig, data["chord_pitch"].T, kind="nearest", fill_value="extrapolate"
    )
    return interp(times_new).T


def process_crema(audio_path, output_dir, output_file_pt):

    if os.path.exists(os.path.join(output_dir, output_file_pt)) == False:
        data_list = []
        labels_list = []
    else:
        test = torch.load(os.path.join(output_dir, output_file_pt))
        data_list = test["data"]
        labels_list = test["labels"]

    out_dict = dict()
    out_dict["crema"] = crema(audio_path)

    label = audio_path.split("/")[-2]

    temp_crema = crema(audio_path)

    os.makedirs(output_dir, exist_ok=True)

    idxs = np.arange(0, temp_crema.shape[0], 8)
    temp_tensor = torch.from_numpy(temp_crema[idxs].T)

    # expanding in the pitch dimension, and adding the feature tensor and its label to the respective lists
    data_list.append(torch.cat((temp_tensor, temp_tensor))[:23].unsqueeze(0))
    labels_list.append(label)

    dataset_dict = {"data": data_list, "labels": labels_list}

    os.makedirs(output_dir, exist_ok=True)

    torch.save(dataset_dict, os.path.join(output_dir, output_file_pt))
    print(torch.load(os.path.join(output_dir, output_file_pt)))


def create_benchmark_ytrue(labels, output_dir, ytrue_labels):
    """
    Function for creating the ground truth file for evaluating models on the Da-TACOS benchmark subset.
    The created ground truth matrix is stored as a .pt file in output_dir directory
    :param labels: labels of the files
    :param output_dir: where to store the ground truth .pt file
    """
    print(f"Creating {ytrue_labels} file.'")
    ytrue = []  # empty list to store ground truth annotations
    for i in range(len(labels)):
        main_label = labels[i]  # label of the ith track in the list
        sub_ytrue = (
            []
        )  # empty list to store ground truth annotations for the ith track in the list
        for j in range(len(labels)):
            if (
                labels[j] == main_label and i != j
            ):  # checking whether the songs have the same label as the ith track
                sub_ytrue.append(1)
            else:
                sub_ytrue.append(0)
        ytrue.append(sub_ytrue)

    # saving the ground truth annotations
    torch.save(torch.Tensor(ytrue), os.path.join(output_dir, ytrue_labels))


# Compute Similarity with Re-move
def compute_similarity_remove(audio1_path, audio2_path, model):
    """
    Compute similarity between two audio files using the re-move model.

    Args:
        audio1_path (str): Path to the first audio file.
        audio2_path (str): Path to the second audio file.
        model: The loaded re-move model.

    Returns:
        float: Similarity score between the two audio files.
    """
    # Preprocess audio files to CSI features
    features1 = preprocess_audio(audio1_path).to(DEVICE)
    features2 = preprocess_audio(audio2_path).to(DEVICE)

    # Pass features through the model to obtain embeddings
    with torch.no_grad():
        embedding1 = model(features1)
        embedding2 = model(features2)

    # Compute cosine similarity
    similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
    return similarity.item()


eval_df = "temp_v0.pt"
output_dir = "temp_dir"

item_list = [
    "../cover500/500/3ahbE6bcVf8.m4a",
    "../cover500/109/642Qxap_c78.m4a",
    "../cover500/123/bMoY5rNBjwk.m4a",
    "../cover500/123/L3xPVGosADQ.m4a",
]

for item in item_list:
    process_crema(item, output_dir, eval_df)

ytrue = torch.load(os.path.join(output_dir, eval_df))["labels"]
create_benchmark_ytrue(ytrue, output_dir, "ytrue_v0.pt")
