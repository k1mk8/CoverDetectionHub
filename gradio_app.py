import gradio as gr
# from audio_utils import preprocess_audio
from models.models import (
    load_bytecover_model,
    load_coverhunter_model,
    compute_similarity_bytecover,
    compute_similarity_coverhunter,
)
from models.utils import compute_similarity, is_cover_song, test_model_on_dataset, evaluate_on_covers80
import librosa
import numpy as np
from scipy.spatial.distance import cosine

from preprocessing import InvalidMediaFileError, validate_audio

# Load ByteCover model
bytecover_model = load_bytecover_model()
coverhunter_model = load_coverhunter_model()

# Gradio Interface for CoverHunter Integration
def gradio_cover_interface(audio1, audio2, model_name, threshold):
    try:
        # Validate and preprocess audio files
        audio1, error1 = validate_audio(audio1)
        if error1:
            return f"Error with Query Song: {error1}", ""

        audio2, error2 = validate_audio(audio2)
        if error2:
            return f"Error with Potential Cover Song: {error2}", ""
    except InvalidMediaFileError as e:
        return str(e), ""

    if model_name == "ByteCover":
        similarity = compute_similarity_bytecover(audio1, audio2, bytecover_model)
    elif model_name == "CoverHunter":
        similarity = compute_similarity_coverhunter(audio1, audio2, coverhunter_model)
    else:
        similarity = compute_similarity(audio1, audio2, model_name)

    # Determine cover or not based on similarity
    result = "Cover" if similarity >= threshold else "Not a Cover"
    return result, f"Similarity Score: {similarity}"

# Gradio app setup for dataset testing
def gradio_test_interface(model_name, dataset, threshold):
    if dataset == "Covers80":
        results = evaluate_on_covers80(model_name, threshold)
    else:
        results = test_model_on_dataset(model_name, dataset)

    summary_table = (
        f"Mean Average Precision (mAP): {results.get('Mean Average Precision (mAP)', 'N/A')}\n"
        f"Precision at 10 (P@10): {results.get('Precision at 10 (P@10)', 'N/A')}\n"
        f"Mean Rank of First Correct Cover (MR1): {results.get('Mean Rank of First Correct Cover (MR1)', 'N/A')}"
    )
    summary_table += f"\nThreshold Used: {threshold}"
    return summary_table

# Example data for Cover Song Identification
examples = [
    ["datasets/example_audio/cicha_noc1.mp3", "datasets/example_audio/cicha_noc2.mp3", "ByteCover", 0.998],
    ["datasets/example_audio/cicha_noc1.mp3", "datasets/example_audio/cicha_noc2.mp3", "CoverHunter", 0.8],
    ["datasets/example_audio/something1.mp3", "datasets/example_audio/something2.mp3", "ByteCover", 0.998],
    ["datasets/example_audio/something1.mp3", "datasets/example_audio/something3.mp3", "CoverHunter", 0.8],
]

# Gradio UI
app1 = gr.Interface(
    fn=gradio_cover_interface,
    inputs=[
        gr.Audio(type="filepath", label="Query Song"),
        gr.Audio(type="filepath", label="Potential Cover Song"),
        gr.Dropdown(choices=["MFCC", "Spectral Centroid", "ByteCover", "CoverHunter"], value="MFCC", label="Choose CSI Model"),
        gr.Slider(minimum=0.5, maximum=1.0, step=0.0001, value=0.99, label="Threshold")
    ],
    outputs=[
        gr.Textbox(label="Result"),
        gr.Textbox(label="Similarity Score")
    ],
    title="Cover Song Identification Hub",
    description="Upload two audio files to check if they are covers of each other. Select the desired CSI model and set the threshold.",
    examples=examples
)

app2 = gr.Interface(
    fn=gradio_test_interface,
    inputs=[
        gr.Dropdown(choices=["ByteCover", "CoverHunter", "MFCC", "Spectral Centroid"], value="Model 1", label="Choose CSI Model"),
        gr.Dropdown(choices=["Dataset A", "Dataset B", "Dataset C", "Covers80"], value="Dataset A", label="Choose Dataset"),
        gr.Slider(minimum=0.5, maximum=1.0, step=0.0001, value=0.99, label="Threshold")
    ],
    outputs=[
        gr.Textbox(label="Summary Metrics")
    ],
    title="Model Testing on Dataset",
    description="Select a CSI model, dataset, and threshold to evaluate performance metrics such as mAP, P@10, and MR1."
)

demo = gr.TabbedInterface([app1, app2], ["Cover Song Identification", "Model Testing"])

if __name__ == "__main__":
    demo.launch()
