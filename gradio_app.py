import gradio as gr
from models.models import load_bytecover_model
from models.utils import is_cover_song, test_model_on_dataset

# Load ByteCover model
bytecover_model = load_bytecover_model()

# Gradio app setup for cover song identification
def gradio_cover_interface(audio1, audio2, model_name):
    result, score = is_cover_song(audio1, audio2, model_name, bytecover_model=bytecover_model)
    return result, f"Similarity Score: {score}"

# Gradio app setup for dataset testing
def gradio_test_interface(model_name, dataset):
    results = test_model_on_dataset(model_name, dataset)
    summary_table = (
        f"Mean Average Precision (mAP): {results['Mean Average Precision (mAP)']}\n"
        f"Precision at 10 (P@10): {results['Precision at 10 (P@10)']}\n"
        f"Mean Rank of First Correct Cover (MR1): {results['Mean Rank of First Correct Cover (MR1)']}"
    )
    return summary_table

# Gradio UI
app1 = gr.Interface(
    fn=gradio_cover_interface,
    inputs=[
        gr.Audio(type="filepath", label="Query Song"),
        gr.Audio(type="filepath", label="Potential Cover Song"),
        gr.Dropdown(choices=["Model 1", "Model 2", "ByteCover"], value="Model 1", label="Choose CSI Model")
    ],
    outputs=[
        gr.Textbox(label="Result"),
        gr.Textbox(label="Similarity Score")
    ],
    title="Cover Song Identification Hub",
    description="Upload two audio files to check if they are covers of each other. Select the desired CSI model from the dropdown."
)

app2 = gr.Interface(
    fn=gradio_test_interface,
    inputs=[
        gr.Dropdown(choices=["Model 1", "Model 2", "ByteCover"], value="Model 1", label="Choose CSI Model"),
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
