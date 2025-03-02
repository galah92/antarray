import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from model_trainer import ArrayPatternCNN
import train
import analyze


def load_model(model_path="array_parameter_model_best.pt", data_dir="array_dataset"):
    """Load a trained model and metadata"""
    # Load metadata to get input dimensions and normalization ranges
    metadata = torch.load(Path(data_dir) / "metadata.pt")
    theta = metadata["theta"]
    xn_range = metadata["xn_range"]
    dx_range = metadata["dx_range"]

    # Initialize model with correct input size
    model = ArrayPatternCNN(len(theta))

    # Load model weights
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, theta, xn_range, dx_range


def predict_parameters(model, pattern, xn_range, dx_range):
    """Predict array parameters from a pattern"""
    # Ensure pattern is a tensor and add batch dimension if needed
    if not isinstance(pattern, torch.Tensor):
        pattern = torch.tensor(pattern, dtype=torch.float32)

    if pattern.dim() == 1:
        pattern = pattern.unsqueeze(0)

    # Get prediction
    with torch.no_grad():
        normalized_pred = model(pattern)

    # Denormalize prediction
    pred_xn = normalized_pred[0, 0].item() * (xn_range[1] - xn_range[0]) + xn_range[0]
    pred_dx = normalized_pred[0, 1].item() * (dx_range[1] - dx_range[0]) + dx_range[0]

    # Round xn to nearest integer
    pred_xn_rounded = round(pred_xn)

    return pred_xn_rounded, pred_dx


def test_with_real_data(
    model_path="array_parameter_model_best.pt",
    data_dir="array_dataset",
    sim_dir=None,
    freq=2.45e9,
):
    """Test the model with real array patterns"""
    if sim_dir is None:
        sim_dir = Path.cwd() / "src" / "sim" / "antenna_array"
    else:
        sim_dir = Path(sim_dir)

    # Load model and metadata
    model, theta, xn_range, dx_range = load_model(model_path, data_dir)

    # Define test configurations
    test_configs = [(1, 60), (2, 70), (4, 80), (8, 65), (16, 75)]

    # Get element data
    elem_E_norm_dbi, elem_Dmax, theta, phi = train.get_elem_data(
        sim_dir, freq, normalize=True
    )

    fig, axes = plt.subplots(len(test_configs), 1, figsize=(10, 4 * len(test_configs)))
    if len(test_configs) == 1:
        axes = [axes]

    for i, (xn, dx) in enumerate(test_configs):
        # Generate array pattern
        array_pattern = train.calc_array_E_norm(
            elem_E_norm_dbi, elem_Dmax, theta, phi, freq, xn, dx=dx, dy=dx
        )

        # Predict parameters
        pred_xn, pred_dx = predict_parameters(model, array_pattern, xn_range, dx_range)

        # Plot result
        ax = axes[i]
        ax.plot(theta * 180 / np.pi, array_pattern)
        ax.set_title(
            f"True: {xn} elements, {dx}mm spacing | Predicted: {pred_xn} elements, {pred_dx:.1f}mm spacing"
        )
        ax.set_xlabel("Theta (degrees)")
        ax.set_ylabel("Directivity (dBi)")
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("model_predictions.png")
    plt.show()


if __name__ == "__main__":
    test_with_real_data()
