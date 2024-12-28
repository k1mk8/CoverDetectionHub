import gradio as gr
from models.models import (
    load_bytecover_model,
    load_coverhunter_model,
    compute_similarity_bytecover,
    compute_similarity_coverhunter,
)
from models.utils import is_cover_song, test_model_on_dataset

# Load ByteCover model
bytecover_model = load_bytecover_model()
coverhunter_model = load_coverhunter_model()

# Gradio Interface for CoverHunter Integration
def gradio_cover_interface(audio1, audio2, model_name):
    if model_name == "ByteCover":
        similarity = compute_similarity_bytecover(audio1, audio2, bytecover_model)
        threshold = 0.99
    elif model_name == "CoverHunter":
        similarity = compute_similarity_coverhunter(audio1, audio2, coverhunter_model)
        threshold = 0.9
    else:
        threshold = 0.9
        result, similarity = is_cover_song(audio1, audio2, model_name)
    
    # Determine cover or not based on similarity
    result = "Cover" if similarity >= threshold else "Not a Cover"
    return result, f"Similarity Score: {similarity:.2f}"

# Gradio app setup for dataset testing
def gradio_test_interface(model_name, dataset):
    results = test_model_on_dataset(model_name, dataset)
    summary_table = (
        f"Mean Average Precision (mAP): {results['Mean Average Precision (mAP)']}\n"
        f"Precision at 10 (P@10): {results['Precision at 10 (P@10)']}\n"
        f"Mean Rank of First Correct Cover (MR1): {results['Mean Rank of First Correct Cover (MR1)']}"
    )
    return summary_table

# Example data for Cover Song Identification
examples = [
    ["datasets/example_audio/cicha_noc1.mp3", "datasets/example_audio/cicha_noc2.mp3", "ByteCover"],
    ["datasets/example_audio/cicha_noc1.mp3", "datasets/example_audio/cicha_noc2.mp3", "CoverHunter"],
    ["datasets/example_audio/something1.mp3", "datasets/example_audio/something2.mp3", "ByteCover"],
    ["datasets/example_audio/something1.mp3", "datasets/example_audio/something3.mp3", "CoverHunter"],
]

# Gradio UI
app1 = gr.Interface(
    fn=gradio_cover_interface,
    inputs=[
        gr.Audio(type="filepath", label="Query Song"),
        gr.Audio(type="filepath", label="Potential Cover Song"),
        gr.Dropdown(choices=["MFCC", "Spectral Centroid", "ByteCover", "CoverHunter"], value="MFCC", label="Choose CSI Model")
    ],
    outputs=[
        gr.Textbox(label="Result"),
        gr.Textbox(label="Similarity Score")
    ],
    title="Cover Song Identification Hub",
    description="Upload two audio files to check if they are covers of each other. Select the desired CSI model from the dropdown.",
    examples=examples
)

app2 = gr.Interface(
    fn=gradio_test_interface,
    inputs=[
        gr.Dropdown(choices=["Model 1", "Model 2", "ByteCover", "CoverHunter"], value="Model 1", label="Choose CSI Model"),
        gr.Dropdown(choices=["Dataset A", "Dataset B", "Dataset C"], value="Dataset A", label="Choose Dataset")
    ],
    outputs=[
        gr.Textbox(label="Summary Metrics")
    ],
    title="Model Testing on Dataset",
    description="Select a CSI model and dataset to evaluate performance metrics such as mAP, P@10, and MR1."
)

app = gr.TabbedInterface([app1, app2], ["Cover Song Identification", "Model Testing"])

if __name__ == "__main__":
    app.launch()
