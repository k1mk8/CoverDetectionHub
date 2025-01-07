from feature_extraction.audio_preprocessing import validate_audio, InvalidMediaFileError
from feature_extraction.feature_extraction import compute_similarity
from csi_models.bytecover_utils import load_bytecover_model, compute_similarity_bytecover
from csi_models.coverhunter_utils import load_coverhunter_model, compute_similarity_coverhunter
from csi_models.lyricover_utils import (
    load_lyricover_model,
    compute_similarity_lyricover
)

from evaluation.covers80_eval import evaluate_on_covers80
from evaluation.abracadabra_eval import evaluate_on_injected_abracadabra


bytecover_model = load_bytecover_model()
coverhunter_model = load_coverhunter_model()
lyricover_model = load_lyricover_model(instrumental_threshold=8)

def gradio_cover_interface(audio1, audio2, model_name, threshold):
    """
    Gradio interface function for checking if two audio files are covers of each other.
    """
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

    # Compute similarity using whichever model is selected
    if model_name == "ByteCover":
        similarity = compute_similarity_bytecover(audio1, audio2, bytecover_model)
    elif model_name == "CoverHunter":
        similarity = compute_similarity_coverhunter(audio1, audio2, coverhunter_model)
    elif model_name == "Lyricover":
        similarity = compute_similarity_lyricover(audio1, audio2, lyricover_model)
    else:
        # For MFCC or Spectral Centroid
        similarity = compute_similarity(audio1, audio2, model_name)

    # Determine cover or not based on similarity & threshold
    result = "Cover" if similarity >= threshold else "Not a Cover"
    return result, f"Similarity Score: {similarity}"

def gradio_test_interface(model_name, dataset):
    """
    Gradio interface function for testing a model on a given dataset (Covers80, etc.).
    """
    if dataset == "Covers80":
        results = evaluate_on_covers80(model_name)
    elif dataset == "Covers80but10":
        results = evaluate_on_covers80(model_name, covers80but10=True)
    elif dataset == "Injected Abracadabra":
        results = evaluate_on_injected_abracadabra(model_name)
    else:
        return "Invalid dataset selected."

    summary_table = (
        f"Mean Average Precision (mAP): {results.get('Mean Average Precision (mAP)', 'N/A')}\n"
        f"Precision at 10 (P@10): {results.get('Precision at 10 (mP@10)', 'N/A')}\n"
        f"Mean Rank of First Correct Cover (MR1): {results.get('Mean Rank of First Correct Cover (mMR1)', 'N/A')}\n"
    )
    return summary_table
