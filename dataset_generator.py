import os
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

import train


def generate_dataset(
    sim_dir=None,
    output_dir="array_dataset",
    xn_range=(1, 64),
    dx_range=(60, 80),
    freq=2.45e9,
    batch_size=32,
):
    """
    Generate a dataset of array patterns with various xn and dx configurations.
    Save the dataset in batches to avoid memory issues.

    Parameters:
        sim_dir: Directory containing simulation files
        output_dir: Directory where dataset will be saved
        xn_range: Range of xn values (min, max)
        dx_range: Range of dx values (min, max)
        freq: Frequency in Hz
        batch_size: Number of samples per batch file
    """
    if sim_dir is None:
        sim_dir = Path.cwd() / "src" / "sim" / "antenna_array"
    else:
        sim_dir = Path(sim_dir)

    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load the element antenna pattern
    elem_E_norm_dbi, elem_Dmax, theta, phi = train.get_elem_data(
        sim_dir, freq, normalize=True
    )

    # Define ranges
    xn_values = range(xn_range[0], xn_range[1] + 1)
    dx_values = range(dx_range[0], dx_range[1] + 1)

    # Calculate total number of samples and batches
    total_samples = len(xn_values) * len(dx_values)
    num_batches = (total_samples + batch_size - 1) // batch_size  # Ceiling division

    print(f"Generating {total_samples} samples across {num_batches} batches")

    # Generate dataset in batches
    sample_idx = 0
    batch_idx = 0
    current_batch_x = []
    current_batch_y = []

    for xn in tqdm(xn_values, desc="Generating dataset"):
        for dx in dx_values:
            # Calculate array pattern
            array_E_norm_dbi = train.calc_array_E_norm(
                elem_E_norm_dbi, elem_Dmax, theta, phi, freq, xn=xn, yn=1, dx=dx, dy=dx
            )

            # Store pattern and labels
            current_batch_x.append(array_E_norm_dbi)
            current_batch_y.append([xn, dx])

            sample_idx += 1

            # Save batch if it's full or this is the last sample
            if len(current_batch_x) == batch_size or sample_idx == total_samples:
                # Convert to tensors
                x_tensor = torch.tensor(np.array(current_batch_x), dtype=torch.float32)
                y_tensor = torch.tensor(np.array(current_batch_y), dtype=torch.float32)

                # Save the batch
                torch.save(
                    {
                        "patterns": x_tensor,
                        "labels": y_tensor,
                    },
                    output_dir / f"array_batch_{batch_idx}.pt",
                )

                # Reset batch data
                current_batch_x = []
                current_batch_y = []
                batch_idx += 1

    # Save the theta values for later reference
    torch.save(
        {
            "theta": torch.tensor(theta),
            "total_samples": total_samples,
            "xn_range": xn_range,
            "dx_range": dx_range,
        },
        output_dir / "metadata.pt",
    )

    print(
        f"Dataset generation complete. {total_samples} samples saved across {batch_idx} batch files."
    )


if __name__ == "__main__":
    generate_dataset(batch_size=64)
