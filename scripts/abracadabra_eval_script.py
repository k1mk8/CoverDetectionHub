


import argparse
import os
import sys
import pandas as pd
import gradio as gr
from tqdm import tqdm
import yaml


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from csi_models.bytecover_utils import compute_similarity_bytecover, load_bytecover_model
from csi_models.coverhunter_utils import compute_similarity_coverhunter, load_coverhunter_model
from csi_models.lyricover_utils import compute_similarity_lyricover, load_lyricover_model
from evaluation.metrics import compute_mean_metrics_for_rankings
from feature_extraction.feature_extraction import compute_similarity


with open("configs/paths.yaml", "r") as f:
    config = yaml.safe_load(f)

INJECTED_ABRACADABRA_DIR = config["injected_abracadabra_dir"]
INJECTED_ABRACADABRA_DATA_DIR = config["injected_abracadabra_data_dir"]
GROUND_TRUTH_FILE = config["injected_abracadabra_ground_truth_file"]
REFERENCE_SONG = config["injected_abracadabra_reference_song"]


def gather_injected_abracadabra_files(dataset_path, ground_truth_file):
    """
    Return (file_path, is_injected) pairs by reading from the injection_list CSV.
    """
    injection_data = pd.read_csv(ground_truth_file)
    # Normalize the path by replacing backslashes
    injection_dict = dict(
        zip(
            injection_data['File'].str.replace("\\", "/"),
            injection_data['Injected'].map(lambda x: x == 'Yes')
        )
    )

    all_audio_files = [
        os.path.join(dataset_path, f)
        for f in os.listdir(dataset_path) if f.endswith(".wav")
    ]

    # Build a list of (path, ground_truth_is_injected)
    files_with_ground_truth = []
    for path in all_audio_files:
        
        rel_path = os.path.relpath(path, start=INJECTED_ABRACADABRA_DIR).replace("\\", "/")

        # Look up in injection_dict
        is_injected = injection_dict.get(rel_path, False)
        files_with_ground_truth.append((path, is_injected))


    return files_with_ground_truth



def compute_rankings_per_song(files_and_labels, similarity_function, progress=gr.Progress()):
    """
    For each song in 'files_and_labels', create a separate ranking 
    based on similarity to other songs.
    
    Returns a list of dictionaries:
    [
      {
        "query_path": ...,
        "query_label": ...,
        "ranking": [
          { "candidate_path": ..., "similarity": float, "ground_truth": bool },
          ...
        ]
      },
      ...
    ]
    """
    rankings_per_query = []
    total_comparisons = len(files_and_labels)
    progress(0, desc="Initializing pairwise comparisons")
    
    for i, (query_path, query_label) in enumerate(tqdm(files_and_labels)):
        # Build a list of comparisons to the reference song
        comparisons = []
        

        sim = similarity_function(query_path, REFERENCE_SONG)
        # print(query_path, query_label)
        gt = query_label
        comparisons.append({
            "candidate_path": query_path,
            "similarity": sim,
            "ground_truth": gt
        })

        # Sort descending by similarity
        comparisons.sort(key=lambda x: x["similarity"], reverse=True)

        # Save the ranking for query_path
        rankings_per_query.append({
            "query_path": query_path,
            "query_label": query_label,
            "ranking": comparisons
        })
        progress((i + 1) / total_comparisons, desc="Evaluating pairs")

    return rankings_per_query


def evaluate_on_injected_abracadabra(
    model_name,
    k=10
):
    """
    Loads the dataset, computes rankings per song,
    and calculates mAP, mP@k, mMR1.
    """
    dataset_path = INJECTED_ABRACADABRA_DATA_DIR
    ground_truth_file = GROUND_TRUTH_FILE
    reference_song = REFERENCE_SONG

    # 2. Prepare similarity_function depending on the model
    if model_name == "ByteCover":
        model = load_bytecover_model()
        def similarity_function(a, b):
            return compute_similarity_bytecover(a, b, model)
    elif model_name == "CoverHunter":
        model = load_coverhunter_model()
        def similarity_function(a, b):
            return compute_similarity_coverhunter(a, b, model)
    elif model_name == "Lyricover":
        model = load_lyricover_model()
        def similarity_function(a, b):
            return compute_similarity_lyricover(a, b, model)
    elif model_name in ["MFCC", "Spectral Centroid"]:
        def similarity_function(a, b):
            return compute_similarity(a, b, model_name)
        model = None
    else:
        raise ValueError("Unsupported model. Choose ByteCover, CoverHunter, Lyricover, MFCC or Spectral Centroid.")

    # 3. Load the list of (file, label)
    files_and_labels = gather_injected_abracadabra_files(dataset_path, ground_truth_file)

    # 4. Ranking for each song
    rankings_per_query = compute_rankings_per_song(files_and_labels, similarity_function)

    # 5. Compute aggregate metrics
    metrics = compute_mean_metrics_for_rankings(rankings_per_query, k=k)
    
    # 6. Return results
    return {
        "Mean Average Precision (mAP)": metrics["mAP"],
        f"Precision at {k} (mP@{k})": metrics["mP@k"],
        "Mean Rank of First Correct Cover (mMR1)": metrics["mMR1"]
    }



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate cover song detection models on the covers80 dataset.")
    
    # Add argument for model selection
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["ByteCover", "CoverHunter", "Lyricover", "MFCC", "Spectral Centroid"],
        help="The name of the model to evaluate (ByteCover, CoverHunter, Lyricover, MFCC, Spectral Centroid)."
    )

    args = parser.parse_args()

    result = evaluate_on_injected_abracadabra(
        model_name=args.model
    )
    print("Evaluation Results:")
    for key, value in result.items():
        print(f"{key}: {value}")